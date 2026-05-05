사용 컨트롤러: forward_position_controller
    note -- servo_node가 내부적으로 Twist 명령을 미세한 Position 값으로 변환하여 송출함.

로봇 IP: 192.168.1.101 / 노트북 IP: 192.168.1.102

Terminal 1 - ur_control.launch.py 실행 후 티칭 펜던트에서 실행 버튼 클릭
    
    ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=192.168.1.101 reverse_ip:=192.168.1.102 launch_rviz:=false

Terminal 2 - ur_moveit.launch.py 실행 시 launch_servo:=true 옵션 넣고 실행
    
    ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3 launch_rviz:=true launch_servo:=true

Terminal 3 - forward_position_controller로 활성화
    
    ros2 control switch_controllers --deactivate forward_velocity_controller --activate forward_position_controller

Terminal 4 - rs_launch.py 실행 (align_depth 활성화 필수)
    
    ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

Terminal 5 - switch_command_type 서비스를 호출하여 command_type: 1로 설정

    # TWIST mode
    ros2 service call /servo_node/switch_command_type moveit_msgs/srv/ServoCommandType "{command_type: 1}"

Terminal 6 - visual_servo_RS.py 실행
    
    python3 visual_servo_RS.py
