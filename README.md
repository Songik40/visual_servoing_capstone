사용 컨트롤러: forward_position_controller
    servo_node가 내부적으로 Twist 명령을 미세한 Position 값으로 변환하여 송출함.

로봇 IP: 192.168.1.101 / 노트북 IP: 192.168.1.102

터미널1 - ur_control.launch.py 실행 후 티칭 펜던트에서 실행 버튼 클릭
터미널2 - ur_moveit.launch.py 실행 시 launch_servo:=true 옵션 넣고 실행
터미널3 - forward_position_controller로 활성화
터미널4 - rs_launch.py 실행 (align_depth 활성화 필수)
터미널5 - switch_command_type 서비스를 호출하여 command_type: 1로 설정
터미널6 - visual_servo_RS.py 실행
