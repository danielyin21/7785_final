#!/usr/bin/env python3
"""
SignClassifier Node using YOLOv5s for Gazebo camera input
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import torch
import numpy as np

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)

class YoloDetector:
    """Wrapper for custom YOLOv5 small model."""
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=model_path, force_reload=False
        ).to(device)
        self.model.eval()
        self.model.conf = 0.2  # confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold

    def predict(self, image: np.ndarray) -> int:
        # BGR image → RGB
        rgb = image[..., ::-1]
        results = self.model(rgb)
        preds = results.xyxy[0]  # (N,6): x1,y1,x2,y2,conf,cls
        if preds.shape[0] == 0:
            return 0
        # pick highest-confidence detection
        confs = preds[:, 4]
        best = int(confs.argmax())
        return int(preds[best, 5])

class SignClassifier(Node):
    """ROS2 node: subscribes to raw Image, predicts sign ID, publishes Int32"""
    def __init__(self):
        super().__init__('sign_classifier')
        self.get_logger().info('SignClassifier: starting')
        self.bridge = CvBridge()
        # load YOLO model
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = YoloDetector('yolov5s_best.pt', dev)

        # subscriber to Gazebo camera
        self.create_subscription(
            Image,
            '/simulated_camera/image_raw',
            self.image_cb,
            10
        )
        # publisher for sign ID
        self.pub = self.create_publisher(Int32, '/sign_id', 10)

    def image_cb(self, msg: Image):
        try:
            # convert ROS Image to OpenCV BGR
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CVBridge error: {e}')
            return
        # predict class
        cls = self.detector.predict(img)
        # publish sign ID
        self.pub.publish(Int32(data=cls))
        self.get_logger().info(f'Predicted sign_id={cls}')

def main():
    rclpy.init()
    node = SignClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
