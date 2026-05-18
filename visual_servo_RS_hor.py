#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, JointState
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math

class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo_node')

        self.color_sub = self.create_subscription(
            CompressedImage, '/camera/camera/color/image_raw/compressed',
            self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        self.publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        self.kp_linear  = 0.002
        self.kp_depth   = 0.0003   # mm → m/s
        self.kp_wrist3  = 0.5
        self.max_linear = 0.1
        self.max_angular = 0.3
        self.search_radius_max = 0.04
        
        # ########## [수정됨: 타겟 및 뎁스 파라미터 변경] 시작 ##########
        # 기존: self.target_object = 'mouse'
        self.target_object = 'bottle'
        self.target_depth = 150.0  # 물병 앞 15cm(150mm)에서 Hovering 정지
        self.blind_forward_count = 0  # Blind Grasp 시 프레임 카운터
        # ########## [수정됨: 타겟 및 뎁스 파라미터 변경] 끝 ##########

        self.cv_depth_image = None

        self.last_bx = 0
        self.last_by = 0
        self.last_x1 = 0
        self.last_y1 = 0
        self.last_x2 = 0
        self.last_y2 = 0

        # 상태 머신
        self.state = 'SEARCHING'
        self.servo_phase = 'WRIST'   # WRIST → X → Y → Z
        self.miss_count = 0
        self.search_angle = 0.0
        self.search_radius = 0.01
        self.hover_stable_count = 0

        # wrist_3 제어 (tool0 프레임에서 angular.z ≈ wrist_3 회전)
        self.q6 = 0.0
        self.wrist3_target = 0.0   # SEARCHING→SERVOING 전환 시점에 캡처
        self.wrist3_thresh = 0.05  # rad (약 3°)

        self.warmup_frames = 10
        self.frame_count = 0

        self.get_logger().info("RealSense D455 비주얼 서보잉 (측면 파지 모드)...")

    def joint_callback(self, msg):
        if 'wrist_3_joint' in msg.name:
            self.q6 = msg.position[msg.name.index('wrist_3_joint')]

    def depth_callback(self, msg):
        self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

    def color_callback(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return
        height, width = cv_image.shape[:2]
        cx, cy = width // 2, height // 2

        results = self.model(cv_image, verbose=False)
        target_found = False
        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'tool0'  # angular.z → wrist_3 직접 대응

        cmd_msg.twist.angular.x = 0.0
        cmd_msg.twist.angular.y = 0.0
        cmd_msg.twist.angular.z = 0.0

        for r in results:
            for box in r.boxes:
                if self.model.names[int(box.cls[0])] == self.target_object:
                    target_found = True
                    self.miss_count = 0
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self.last_x1, self.last_y1 = x1, y1
                    self.last_x2, self.last_y2 = x2, y2
                    self.last_bx, self.last_by = (x1+x2)//2, (y1+y2)//2
                    break
            if target_found:
                break

        if not target_found:
            self.miss_count += 1
            if self.miss_count > 30 and self.state == 'SERVOING':
                self.state = 'SEARCHING'
                self.search_angle = 0.0
                self.search_radius = 0.01
                self.get_logger().warn("타겟 상실... 탐색 시작")

        # ── [상태 1] SEARCHING ──────────────────────────────────────────
        if self.state == 'SEARCHING':
            if target_found:
                self.wrist3_target = self.q6  # 현재 wrist_3 각도를 목표로 고정
                self.servo_phase = 'WRIST'
                self.state = 'SERVOING'
                self.get_logger().info(f"타겟 포착... wrist3 고정({self.wrist3_target:.3f} rad) 후 X→Y→Z")
            else:
                # ########## [수정됨: 나선형 탐색 ➡️ 좌우(Pan) 스윕으로 변경] 시작 ##########
                omega = 1.0
                self.search_angle += omega / 30.0
                
                # Z축과 X축은 고정하고, Y축(툴 좌표계 기준 좌우)으로만 흔들기
                cmd_msg.twist.linear.x = 0.0
                cmd_msg.twist.linear.y = 0.04 * math.sin(self.search_angle)
                cmd_msg.twist.linear.z = 0.0
                
                cv2.putText(cv_image, "Mode: SEARCHING (PAN)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                # ########## [수정됨: 나선형 탐색 ➡️ 좌우(Pan) 스윕으로 변경] 끝 ##########

        # ── [상태 2] SERVOING ───────────────────────────────────────────
        elif self.state == 'SERVOING':
            error_x = self.last_bx - cx
            error_y = self.last_by - cy
            wrist3_err = self.q6 - self.wrist3_target

            cv2.rectangle(cv_image, (self.last_x1, self.last_y1), (self.last_x2, self.last_y2), (0,255,0), 2)
            cv2.line(cv_image, (cx, cy), (self.last_bx, self.last_by), (0,255,255), 2)
            cv2.putText(cv_image, f"Mode: SERVOING [{self.servo_phase}]", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(cv_image, f"err_x:{error_x:+d}px  err_y:{error_y:+d}px", (50,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # [페이즈 0] WRIST — wrist_3 정렬, linear 정지
            if self.servo_phase == 'WRIST':
                cmd_msg.twist.angular.z = float(np.clip(
                    -self.kp_wrist3 * wrist3_err, -self.max_angular, self.max_angular))
                cv2.putText(cv_image, f"wrist3_err: {wrist3_err:+.3f} rad", (50,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
                if abs(wrist3_err) < self.wrist3_thresh:
                    self.servo_phase = 'X'
                    self.get_logger().info("wrist3 고정 완료 → X 제어 시작")

            # [페이즈 1] X — pixel_x → linear.y (tool0에서 linear.x가 카메라 상하에 대응)
            elif self.servo_phase == 'X':
                cmd_msg.twist.linear.y = float(error_x) * self.kp_linear
                if abs(error_x) < 5:
                    self.servo_phase = 'Y'
                    self.get_logger().info("X 정렬 완료 → Y 제어 시작")

            # [페이즈 2] Y — pixel_y → linear.x
            elif self.servo_phase == 'Y':
                cmd_msg.twist.linear.x = -float(error_y) * self.kp_linear
                if abs(error_y) < 5:
                    self.servo_phase = 'Z'
                    self.get_logger().info("Y 정렬 완료 → Z(depth) 제어 시작")

            # [페이즈 3] Z — depth만 제어
            elif self.servo_phase == 'Z':
                depth_aligned = False
                if self.cv_depth_image is not None and 0 <= self.last_by < height and 0 <= self.last_bx < width:
                    distance_mm = self.cv_depth_image[self.last_by, self.last_bx]
                    if distance_mm > 0:
                        # ########## [수정됨: 목표 뎁스를 350.0에서 self.target_depth로 변경] 시작 ##########
                        depth_error = float(distance_mm) - self.target_depth
                        cmd_msg.twist.linear.z = depth_error * self.kp_depth
                        depth_aligned = abs(depth_error) < 8.0
                        cv2.putText(cv_image, f"Z:{distance_mm}mm  depth_err:{depth_error:+.0f}mm", (50,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
                        # ########## [수정됨: 목표 뎁스를 350.0에서 self.target_depth로 변경] 끝 ##########

                if abs(error_x) < 8 and abs(error_y) < 8 and depth_aligned:
                    self.hover_stable_count += 1
                else:
                    self.hover_stable_count = 0
                cv2.putText(cv_image, f"stable:{self.hover_stable_count}/10", (50,165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

                if self.hover_stable_count >= 10:
                    self.hover_stable_count = 0
                    self.state = 'HOVERING'
                    self.get_logger().info("정렬 성공... HOVERING 및 직진 파지 진입")

        # ── [상태 3 & 4] HOVERING 및 BLIND GRASP ────────────────────────
        # ########## [수정됨: HOVERING 이후 오픈루프 직진 파지 로직 추가] 시작 ##########
        elif self.state == 'HOVERING':
            self.blind_forward_count += 1
            
            # 약 2초(60프레임) 동안 5cm/s 속도로 Z축 직진 (물병 몸통으로 전진)
            if self.blind_forward_count < 60:
                cmd_msg.twist.linear.x = 0.0
                cmd_msg.twist.linear.y = 0.0
                cmd_msg.twist.linear.z = 0.05  # 앞으로 천천히 밀어넣기
                cv2.putText(cv_image, "Mode: BLIND FORWARD", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                # 전진 완료 후 완전 정지 상태로 전환
                cmd_msg.twist.linear.x = 0.0
                cmd_msg.twist.linear.y = 0.0
                cmd_msg.twist.linear.z = 0.0
                self.state = 'GRASPING'
                cv2.putText(cv_image, "Mode: GRASPING (STOP)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                self.get_logger().info("★★★ 직진 완료! 그리퍼를 닫으십시오 ★★★")

        elif self.state == 'GRASPING':
            # 정지 상태 유지 (I/O 컨트롤러로 그리퍼 닫기 명령을 내릴 타이밍)
            cmd_msg.twist.linear.x = 0.0
            cmd_msg.twist.linear.y = 0.0
            cmd_msg.twist.linear.z = 0.0
            cv2.putText(cv_image, "Mode: GRASPING (STOP)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        # ########## [수정됨: HOVERING 이후 오픈루프 직진 파지 로직 추가] 끝 ##########

        # warmup + 안전 가드 + publish
        self.frame_count += 1
        if self.frame_count <= self.warmup_frames:
            cv2.putText(cv_image, f"Warmup {self.frame_count}/{self.warmup_frames}",
                        (50, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        else:
            lin = cmd_msg.twist.linear
            ang = cmd_msg.twist.angular
            vals = [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
            if not all(np.isfinite(v) for v in vals):
                self.get_logger().warn(f"Non-finite cmd 차단: {vals}")
            else:
                lin.x = float(np.clip(lin.x, -self.max_linear, self.max_linear))
                lin.y = float(np.clip(lin.y, -self.max_linear, self.max_linear))
                lin.z = float(np.clip(lin.z, -self.max_linear, self.max_linear))
                self.publisher_.publish(cmd_msg)

        cv2.imshow("RealSense VLA", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
