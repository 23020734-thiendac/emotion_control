"""Map a stable facial-emotion label to a short, safe TurtleBot action."""
from __future__ import annotations

import math
import time
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class EmotionBehaviorNode(Node):
    """Run one action per emotion transition and always provide a stop path.

    The recognizer publishes a stable label, rather than every noisy webcam
    frame.  Repeated labels do not retrigger the action, which prevents a
    smiling face from making the robot drive forever.
    """

    def __init__(self) -> None:
        super().__init__("emotion_behavior_node")
        self.declare_parameter("linear_speed", 0.16)
        self.declare_parameter("angular_speed", 0.55)
        self.declare_parameter("action_duration_sec", 2.5)
        self.declare_parameter("reverse_duration_sec", 1.0)
        self.declare_parameter("forward_duration_sec", 1.0)
        self.declare_parameter("neutral_stop_sec", 0.2)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.action_duration = float(self.get_parameter("action_duration_sec").value)
        self.reverse_duration = float(self.get_parameter("reverse_duration_sec").value)
        self.forward_duration = float(self.get_parameter("forward_duration_sec").value)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/emotion_behavior_status", 10)
        self.emotion_sub = self.create_subscription(
            String, "/facial_emotion", self.on_emotion, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.on_odom, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.last_label = "Unknown"
        self.active_label: Optional[str] = None
        self.action_started = 0.0
        self.action_timeout = 0.0
        self.yaw: Optional[float] = None
        self.spin_start_yaw: Optional[float] = None
        self.spin_turned = 0.0
        self.last_action_status = "Neutral - robot stopped"
        self.publish_stop()
        self.get_logger().info(
            "Emotion behavior san sang: Happy/Sad/Angry/Surprise/Fear -> /cmd_vel."
        )

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(sin_yaw, cos_yaw)

    def on_odom(self, message: Odometry) -> None:
        new_yaw = self._yaw_from_quaternion(message.pose.pose.orientation)
        if self.yaw is not None and self.active_label == "Surprise":
            delta = math.atan2(math.sin(new_yaw - self.yaw), math.cos(new_yaw - self.yaw))
            self.spin_turned += abs(delta)
        self.yaw = new_yaw
        if self.active_label == "Surprise" and self.spin_start_yaw is None:
            self.spin_start_yaw = new_yaw

    def on_emotion(self, message: String) -> None:
        label = message.data.strip().title()
        if label not in {"Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear"}:
            label = "Neutral"

        if label == "Neutral":
            self.last_label = label
            self.active_label = None
            self.publish_stop()
            self.publish_status("Neutral - robot stopped")
            return

        # A stable label is an edge-triggered command.  Holding the same
        # expression does not repeatedly retrigger the robot.
        if label == self.last_label:
            return
        self.last_label = label
        self.start_action(label)

    def start_action(self, label: str) -> None:
        now = time.monotonic()
        self.active_label = label
        self.action_started = now
        self.spin_start_yaw = self.yaw if label == "Surprise" else None
        self.spin_turned = 0.0
        if label in ("Happy", "Sad"):
            self.action_timeout = self.action_duration
            text = f"{label}: curved motion"
        elif label == "Angry":
            self.action_timeout = self.reverse_duration
            text = "Angry: reverse lightly"
        elif label == "Fear":
            self.action_timeout = self.forward_duration
            text = "Fear: move forward lightly"
        else:
            self.action_timeout = 15.0
            text = "Surprise: rotate one full turn"
        self.publish_status(text)
        self.get_logger().info(text)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_status(self, text: str) -> None:
        if text == self.last_action_status:
            return
        self.last_action_status = text
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def finish_action(self) -> None:
        label = self.active_label or "Unknown"
        self.active_label = None
        self.publish_stop()
        self.publish_status(f"{label}: completed; robot stopped")

    def control_loop(self) -> None:
        if self.active_label is None:
            return

        elapsed = time.monotonic() - self.action_started
        twist = Twist()
        if self.active_label == "Happy":
            twist.linear.x = self.linear_speed
            twist.angular.z = abs(self.angular_speed)
        elif self.active_label == "Sad":
            twist.linear.x = self.linear_speed
            twist.angular.z = -abs(self.angular_speed)
        elif self.active_label == "Angry":
            twist.linear.x = -abs(self.linear_speed) * 0.75
        elif self.active_label == "Fear":
            twist.linear.x = abs(self.linear_speed) * 0.75
        elif self.active_label == "Surprise":
            twist.angular.z = abs(self.angular_speed)
            # Odometry is the authoritative completion condition.  The
            # timeout is only a safety fallback if odometry disappears.
            if self.spin_turned >= 2.0 * math.pi:
                self.finish_action()
                return

        if elapsed >= self.action_timeout:
            self.finish_action()
            return
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.publish_stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmotionBehaviorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
