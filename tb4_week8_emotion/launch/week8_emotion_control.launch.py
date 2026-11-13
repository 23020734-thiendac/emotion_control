"""Start the Small House simulation and Week 8 facial-emotion control."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import ReplaceString


def generate_launch_description():
    week8_share = get_package_share_directory("tb4_week8_emotion")
    week7_share = get_package_share_directory("tb4_week7")
    week6_share = get_package_share_directory("tb4_week6")
    navigation_share = get_package_share_directory("turtlebot4_navigation")

    map_file = Path(week7_share) / "maps" / "small_house_slam.yaml"
    zones_file = Path(week7_share) / "config" / "zones.yaml"
    simulation_launch = PathJoinSubstitution([
        week6_share, "launch", "week6_small_house_sim.launch.py"
    ])
    localization_launch = PathJoinSubstitution([
        navigation_share, "launch", "localization.launch.py"
    ])
    nav2_launch = PathJoinSubstitution([
        navigation_share, "launch", "nav2.launch.py"
    ])
    rviz_config = PathJoinSubstitution([
        week8_share, "rviz", "emotion_control.rviz"
    ])

    nav2_params = ReplaceString(
        source_file=PathJoinSubstitution([
            navigation_share, "config", "nav2.yaml"
        ]),
        replacements={
            "max_vel_x: 0.26": "max_vel_x: 0.35",
            "max_speed_xy: 0.26": "max_speed_xy: 0.35",
            "max_vel_theta: 1.0": "max_vel_theta: 1.20",
        },
    )

    args = [
        DeclareLaunchArgument("ros_domain_id", default_value="68"),
        DeclareLaunchArgument("model", default_value="standard"),
        # Spawn in the open central area of Small House, not in Kitchen.
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("emotion_backend", default_value="auto"),
        DeclareLaunchArgument("show_window", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("publish_image", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("draw_landmarks", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("show_debug_cues", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("show_rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("spawn_delay_sec", default_value="12.0"),
        DeclareLaunchArgument("localization_delay_sec", default_value="38.0"),
        DeclareLaunchArgument("nav2_delay_sec", default_value="48.0"),
        DeclareLaunchArgument("initial_pose_delay_sec", default_value="54.0"),
        DeclareLaunchArgument("semantic_delay_sec", default_value="54.0"),
        DeclareLaunchArgument("rviz_delay_sec", default_value="72.0"),
    ]

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulation_launch),
        launch_arguments={
            "ros_domain_id": LaunchConfiguration("ros_domain_id"),
            "model": LaunchConfiguration("model"),
            "rviz": "false",
            "spawn_delay_sec": LaunchConfiguration("spawn_delay_sec"),
            "x": LaunchConfiguration("x"),
            "y": LaunchConfiguration("y"),
            "z": LaunchConfiguration("z"),
            "yaw": LaunchConfiguration("yaw"),
        }.items(),
    )

    localization = TimerAction(
        period=LaunchConfiguration("localization_delay_sec"),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(localization_launch),
            launch_arguments={"map": str(map_file), "use_sim_time": "true"}.items(),
        )],
    )

    navigation = TimerAction(
        period=LaunchConfiguration("nav2_delay_sec"),
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                "use_sim_time": "true",
                "params_file": nav2_params,
            }.items(),
        )],
    )

    initial_pose = TimerAction(
        period=LaunchConfiguration("initial_pose_delay_sec"),
        actions=[Node(
            package="tb4_week7",
            executable="initial_pose_publisher",
            name="week8_initial_pose_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "x": ParameterValue(LaunchConfiguration("x"), value_type=float),
                "y": ParameterValue(LaunchConfiguration("y"), value_type=float),
                "yaw": ParameterValue(LaunchConfiguration("yaw"), value_type=float),
            }],
        )],
    )

    semantic = TimerAction(
        period=LaunchConfiguration("semantic_delay_sec"),
        actions=[Node(
            package="tb4_week7",
            executable="map_segmentation_node",
            name="week8_semantic_map_node",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "map_yaml": str(map_file),
                "zones_yaml": str(zones_file),
            }],
        )],
    )

    emotion = Node(
        package="tb4_week8_emotion",
        executable="emotion_recognition_node",
        name="emotion_recognition_node",
        output="screen",
        parameters=[{
            "camera_index": ParameterValue(
                LaunchConfiguration("camera_index"), value_type=int),
            "backend": LaunchConfiguration("emotion_backend"),
            "show_window": ParameterValue(
                LaunchConfiguration("show_window"), value_type=bool),
            "publish_image": ParameterValue(
                LaunchConfiguration("publish_image"), value_type=bool),
            "draw_landmarks": ParameterValue(
                LaunchConfiguration("draw_landmarks"), value_type=bool),
            "show_debug_cues": ParameterValue(
                LaunchConfiguration("show_debug_cues"), value_type=bool),
            "use_sim_time": False,
        }],
    )

    behavior = Node(
        package="tb4_week8_emotion",
        executable="emotion_behavior_node",
        name="emotion_behavior_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    rviz = TimerAction(
        period=LaunchConfiguration("rviz_delay_sec"),
        actions=[Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_week8",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("show_rviz")),
        )],
    )

    return LaunchDescription(args + [
        simulation,
        localization,
        navigation,
        initial_pose,
        semantic,
        emotion,
        behavior,
        rviz,
    ])
