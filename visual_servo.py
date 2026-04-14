#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo_node')
        
        # RGB 컬러 토픽 구독 
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )
        
        # Depth(거리) 토픽 구독 
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        
        # 3. 로봇팔 제어 토픽 퍼블리셔
        self.publisher_ = self.create_publisher(Twist, '/arm_controller/twist_cmd', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') 
        
        self.kp_x = 0.002
        self.kp_y = 0.002
        self.target_object = 'keyboard' # 테스트용 타겟
        
        # 뎁스 이미지 저장 변수
        self.cv_depth_image = None 

        self.get_logger().info("Eye-in-Hand 비주얼 서보잉 가동")

    # 뎁스 이미지  call_back 함수
    def depth_callback(self, msg):
        # 뎁스 데이터는 보통 16비트 정수(16UC1, 단위: mm)
        self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
    # rgb이미지 call_back 함수
    def color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width = cv_image.shape[:2]
        cx, cy = width // 2, height // 2

        results = self.model(cv_image, verbose=False)
        target_found = False
        cmd_msg = Twist()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                if cls_name == self.target_object:
                    target_found = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bx = (x1 + x2) // 2
                    by = (y1 + y2) // 2
                    
                    error_x = bx - cx
                    error_y = by - cy
                    
                    cmd_msg.angular.z = -float(error_x) * self.kp_x  
                    cmd_msg.angular.y = float(error_y) * self.kp_y   
                    
                    # 화면 피드백 그리기
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(cv_image, (bx, by), 5, (0, 0, 255), -1) 
                    cv2.circle(cv_image, (cx, cy), 5, (255, 0, 0), -1) 
                    cv2.line(cv_image, (cx, cy), (bx, by), (0, 255, 255), 2)
                    
                    # 뎁스 이미지에서 타겟 중앙점의 거리(Z) 값 추출하기
                    distance_mm = 0
                    if self.cv_depth_image is not None:
                        # 좌표가 뎁스 이미지 해상도를 벗어나지 않도록 안전장치
                        if 0 <= by < self.cv_depth_image.shape[0] and 0 <= bx < self.cv_depth_image.shape[1]:
                            
                            distance_mm = self.cv_depth_image[by, bx]
                            
                            if distance_mm > 0: # 0이면 측정 실패
                                
                                text = f"Dist: {distance_mm} mm"
                                cv2.putText(cv_image, text, (bx + 10, by - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                self.get_logger().info(f"타겟 확인... 오차 X:{error_x} Y:{error_y} | 거리 Z:{distance_mm}mm")
                            else:
                                cv2.putText(cv_image, "Dist: Too Close/Far", (bx + 10, by - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    break 
            
            if target_found:
                break

        if not target_found:
            cmd_msg.angular.z = 0.0
            cmd_msg.angular.y = 0.0

        self.publisher_.publish(cmd_msg)
        
        cv2.imshow("Eye-in-Hand Visual Servoing", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
