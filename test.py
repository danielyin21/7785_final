#!/usr/bin/env python3
"""
RotateNavigator with YOLO-based sign detection
"""
import rclpy
import math
import time
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)

# Sign IDs matching your YOLO model's classes
LEFT, RIGHT, DNE, STOP, GOAL = 1, 2, 3, 4, 5
VALID_SIGNS = {LEFT, RIGHT, DNE, STOP, GOAL}

class YoloDetector:
    """Wrapper around a YOLOv5 custom model."""
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=model_path, force_reload=True
        ).to(device)
        self.model.eval()
        # detection thresholds (tune if needed)
        self.model.conf = 0.2
        self.model.iou  = 0.45

    def predict(self, image) -> int:
        # image: OpenCV BGR array
        rgb = image[..., ::-1]
        results = self.model(rgb)
        preds = results.xyxy[0]  # (N,6): x1,y1,x2,y2,conf,cls
        if preds.shape[0] == 0:
            return 0
        confs = preds[:, 4]
        best_idx = int(confs.argmax())
        return int(preds[best_idx, 5])

class RotateNavigator(Node):
    def __init__(self):
        super().__init__('rotate_navigator')
        self.state = 'WAIT_WALL'
        self.sign = None
        self.yaw = 0.0
        self.yaw_orig = None
        self.target_yaw = None
        self.shake_deg = math.radians(30)
        self.curr_img = None

        # YOLO detector + image bridge
        self.bridge = CvBridge()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = YoloDetector('yolov5s_best.pt', device)

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rot_pub = self.create_publisher(Bool, '/rotating', 10)
        self.create_subscription(Bool, '/arrive_wall', self.wall_cb, 10)
        # Subscribe to camera for sign detection
        self.create_subscription(Image, '/simulated_camera/image_raw', self.image_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # Timer for main loop
        self.create_timer(0.02, self.tick)

    def image_cb(self, msg: Image):
        try:
            self.curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")

    def yaw_from_odom(self, q):
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny, cosy)

    def odom_cb(self, msg: Odometry):
        self.yaw = self.yaw_from_odom(msg.pose.pose.orientation)

    def wall_cb(self, msg: Bool):
        if msg.data and self.state == 'WAIT_WALL':
            self.yaw_orig = self.yaw
            self.sign = None
            self.state = 'DECIDE'
            self.rot_pub.publish(Bool(data=True))

    def ang_err(self, tgt):
        return (tgt - self.yaw + math.pi) % (2*math.pi) - math.pi

    def send_omega(self, omega):
        tw = Twist()
        tw.angular.z = omega
        self.cmd_pub.publish(tw)

    def reach_target(self):
        err = abs(self.ang_err(self.target_yaw))
        if err < math.radians(1.5):
            self.cmd_pub.publish(Twist())
            return True
        omega = self.kp * (self.target_yaw - self.yaw)
        self.send_omega(max(-self.out_lim, min(self.out_lim, omega)))
        return False

    def tick(self):
        if self.state == 'DECIDE':
            # Try to detect sign from current image
            if self.curr_img is not None:
                cls = self.detector.predict(self.curr_img)
                if cls in VALID_SIGNS:
                    self.sign = cls
                    # proceed to sign-based rotation
                    self.prepare_sign_rotation()
            return

        if self.state in ('SHAKE_LEFT', 'SHAKE_RIGHT', 'SHAKE_BACK', 'SIGN_ROTATE'):
            if self.reach_target():
                # handle next state transitions
                if self.state == 'SHAKE_LEFT':
                    self.target_yaw = (self.yaw_orig - self.shake_deg) % (2*math.pi)
                    self.state = 'SHAKE_RIGHT'
                elif self.state == 'SHAKE_RIGHT':
                    self.target_yaw = self.yaw_orig
                    self.state = 'SHAKE_BACK'
                elif self.state == 'SHAKE_BACK':
                    if self.sign:
                        self.prepare_sign_rotation()
                    else:
                        self.rot_pub.publish(Bool(data=False))
                        self.state = 'WAIT_WALL'
                elif self.state == 'SIGN_ROTATE':
                    self.rot_pub.publish(Bool(data=False))
                    self.state = 'WAIT_WALL'
            return

    def prepare_sign_rotation(self):
        # Map sign to yaw offset
        offset = 0.0
        if self.sign == LEFT:   offset =  math.pi/2
        elif self.sign == RIGHT:offset = -math.pi/2
        elif self.sign in (DNE, STOP): offset = math.pi
        elif self.sign == GOAL:
            self.get_logger().info('Goal reached!')
            self.cmd_pub.publish(Twist())
            self.rot_pub.publish(Bool(data=False))
            self.state = 'WAIT_WALL'
            return

        self.target_yaw = (self.yaw_orig + offset) % (2*math.pi)
        self.state = 'SIGN_ROTATE'

def main():
    rclpy.init()
    node = RotateNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
