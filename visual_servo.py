#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import math
import time

class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo_node')
        
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/arm_controller/twist_cmd', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        
        # 제어 게인값 (선속도 전용)
        self.kp_linear = 0.0015
        self.target_object = 'cell phone' # 타겟 변경
        
        self.cv_depth_image = None 
        
        # State Machine 변수들
        self.state = 'SEARCHING' # 초기 상태: 탐색
        self.miss_count = 0      # 놓친 프레임 수
        
        # 나선형 탐색(Spiral Search) 변수
        self.search_angle = 0.0
        self.search_radius = 0.01 # 초기 나선 반경
        
        self.get_logger().info("VLA 비주얼 서보잉...")

    def depth_callback(self, msg):
        self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
    def color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width = cv_image.shape[:2]
        cx, cy = width // 2, height // 2

        results = self.model(cv_image, verbose=False)
        target_found = False
        cmd_msg = Twist()
        
        # 타겟 찾기 루프
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_name = self.model.names[int(box.cls[0])]
                if cls_name == self.target_object:
                    target_found = True
                    self.miss_count = 0 # 찾으면 놓친 카운트 초기화
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bx, by = (x1 + x2) // 2, (y1 + y2) // 2
                    break 
            if target_found: break

        # 타겟을 놓쳤을 때의 카운트 로직
        if not target_found:
            self.miss_count += 1
            if self.miss_count > 30 and self.state == 'SERVOING':
                self.state = 'SEARCHING' # 30프레임(약 1초) 이상 안 보이면 다시 수색 작전 돌입
                self.get_logger().warn("타겟 상실... Spiral Search mode로 전환")

        # State에 따른 행동 결정
        
        # 1. Spiral Search
        if self.state == 'SEARCHING':
            if target_found:
                self.state = 'SERVOING' # 타겟 포착
                self.get_logger().info("타겟 포착... 비주얼 서보잉 실행")
            else:
                # 삼각함수를 이용해 X, Y축으로 나선형(원) 평행 이동 속도 생성
                self.search_angle += 0.1
                self.search_radius += 0.0005 # 점점 원이 커지도록
                
                cmd_msg.linear.x = self.search_radius * math.cos(self.search_angle)
                cmd_msg.linear.y = self.search_radius * math.sin(self.search_angle)
                
                cv2.putText(cv_image, "Mode: SEARCHING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # 2. Visual Servoing
        elif self.state == 'SERVOING':
            error_x = bx - cx
            error_y = by - cy
            
            # 오차를 평행 이동(선속도)으로 변환
            cmd_msg.linear.x = -float(error_x) * self.kp_linear
            cmd_msg.linear.y = float(error_y) * self.kp_linear
            
            # 화면 피드백
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(cv_image, (cx, cy), (bx, by), (0, 255, 255), 2)
            cv2.putText(cv_image, "Mode: SERVOING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 호버링 조건 체크: 뎁스 값이 25cm(250mm) 근처이고, X/Y 오차가 거의 없을 때
            distance_mm = 0
            if self.cv_depth_image is not None and 0 <= by < height and 0 <= bx < width:
                distance_mm = self.cv_depth_image[by, bx]
                
                if distance_mm > 0:
                    cv2.putText(cv_image, f"Z: {distance_mm}mm", (bx+10, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    # 🎯 완벽 정렬 확인! (오차 15픽셀 이내, 고도 260mm 이하)
                    if abs(error_x) < 15 and abs(error_y) < 15 and distance_mm <= 260:
                        self.state = 'HOVERING'
                        self.get_logger().info("정렬 완료... 파지(Drop) 준비")

        # 3. Blind Drop
        elif self.state == 'HOVERING':
            cmd_msg.linear.x = 0.0
            cmd_msg.linear.y = 0.0
            cmd_msg.linear.z = -0.05 # 아래로 천천히 하강
            
            cv2.putText(cv_image, "Mode: BLIND DROP (GRASPING)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # (이후 일정 고도 이하로 내려가면 그리퍼를 닫는 로직)

        # 명령 하달 및 화면 출력
        self.publisher_.publish(cmd_msg)
        cv2.imshow("Eye-in-Hand VLA Camera", cv_image)
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
