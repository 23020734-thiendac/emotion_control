# TB4 Week 8 - Nhận diện cảm xúc bằng MediaPipe

Package này nhận diện khuôn mặt qua webcam, phân loại cảm xúc và điều khiển
TurtleBot 4 trong mô phỏng AWS RoboMaker Small House.

## Phương pháp nhận diện

Node sử dụng MediaPipe Face Landmarker với 478 điểm khuôn mặt và Face
Blendshapes. Các đặc trưng được kết hợp gồm:

- vị trí và độ mở của hai mắt;
- độ mở, hình chữ O, độ cong và độ lệch của miệng;
- nhe/nghiến răng;
- chuyển động lông mày và má;
- bàn tay đưa gần mặt để hỗ trợ nhận diện Fear;
- bỏ phiếu nhiều frame để loại nhiễu webcam.

Ánh xạ tín hiệu chính:

| Nhãn | Dấu hiệu ưu tiên |
|---|---|
| Happy | khóe miệng kéo lên, mouthSmile, có thể nhìn thấy răng |
| Sad | mắt nhắm/nhíu, khóe miệng kéo xuống hoặc môi mếu |
| Angry | mắt mở to, lông mày căng, nhe/nghiến răng, miệng lệch |
| Surprise | miệng mở tròn hình chữ O, hàm hạ xuống, lông mày nâng |
| Fear | tay che/sát vùng miệng kết hợp mắt mở và nét mặt căng |
| Neutral | không có tín hiệu đủ mạnh |

Landmark chỉ được dùng để tính đặc trưng và có thể bật/tắt điểm chấm trên ảnh
bằng draw_landmarks:=true/false. Nó không phải là nhãn cảm xúc riêng.

## Cài đặt và build

~~~bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
python3 -m pip install --user mediapipe

colcon build --symlink-install \
  --base-paths src/TuongTacNguoi_Robot \
  --packages-select tb4_week8_emotion

source install/setup.bash
~~~

Hai model MediaPipe nằm trong thư mục models/ và được cài cùng package:

- face_landmarker.task
- hand_landmarker.task

## Chạy nhận diện webcam

Chỉ chạy node webcam:

~~~bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run tb4_week8_emotion emotion_recognition_node --ros-args \
  -p camera_index:=0 \
  -p show_window:=true \
  -p draw_landmarks:=true
~~~

Nếu muốn ảnh sạch, không hiện điểm landmark:

~~~bash
ros2 run tb4_week8_emotion emotion_recognition_node --ros-args \
  -p draw_landmarks:=false
~~~

Nếu webcam là thiết bị số 1, đổi camera_index:=1.

## Chạy đầy đủ Small House và TurtleBot 4

~~~bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=68

ros2 launch tb4_week8_emotion week8_emotion_control.launch.py \
  camera_index:=0 \
  emotion_backend:=mediapipe \
  draw_landmarks:=true \
  show_window:=true \
  show_rviz:=true
~~~

Launch sử dụng bản đồ Small House của tuần 7, khởi động localization, Nav2,
semantic map, RViz và node điều khiển robot của tuần 8.

## Topic chính

| Topic | Kiểu | Ý nghĩa |
|---|---|---|
| /facial_emotion | std_msgs/msg/String | Nhãn ổn định |
| /facial_emotion_confidence | std_msgs/msg/Float32 | Độ tin cậy |
| /facial_emotion/image | sensor_msgs/msg/Image | Ảnh webcam có khung, nhãn và cue |
| /facial_emotion_status | std_msgs/msg/String | Trạng thái webcam/model |
| /emotion_behavior_status | std_msgs/msg/String | Hành vi hiện tại |
| /cmd_vel | geometry_msgs/msg/Twist | Vận tốc TurtleBot 4 |

Theo dõi nhãn:

~~~bash
ros2 topic echo /facial_emotion
ros2 topic echo /facial_emotion_confidence
~~~

## Cấu trúc package

~~~text
tb4_week8_emotion/
├── models/
│   ├── face_landmarker.task
│   └── hand_landmarker.task
├── launch/
│   └── week8_emotion_control.launch.py
├── rviz/
│   └── emotion_control.rviz
├── resource/
│   └── tb4_week8_emotion
├── tb4_week8_emotion/
│   ├── __init__.py
│   ├── emotion_recognition_node.py
│   └── emotion_behavior_node.py
├── package.xml
├── setup.cfg
├── setup.py
└── README.md
~~~

## Lưu ý để nhận diện ổn định

- Đặt mặt cách webcam khoảng 40-80 cm, đủ sáng và không ngược sáng.
- Giữ toàn bộ khuôn mặt trong khung hình.
- Mỗi biểu cảm nên giữ khoảng 1-2 giây để bộ lọc nhiều frame xác nhận.
- Với Angry, cần để camera thấy mắt mở to và phần răng nhe/nghiến.
- Với Surprise, tạo miệng tròn chữ O rõ ràng.
- Với Sad, nhắm/nhíu mắt và làm môi mếu xuống.
- Không có mặt hoặc tín hiệu yếu thì robot nhận Unknown/Neutral và dừng.
