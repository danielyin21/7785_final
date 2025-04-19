# classification_node.py
#!/usr/bin/env python3
import logging
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import numpy as np
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        logger.info(f"Loading YOLO model from {model_path} on {device}")
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=model_path, force_reload=False
        ).to(device)
        self.model.eval()
        self.model.conf = 0.2
        self.model.iou  = 0.45

    def predict(self, image: np.ndarray) -> int:
        rgb = image[..., ::-1]
        results = self.model(rgb)
        preds = results.xyxy[0]
        if preds.shape[0] == 0:
            return 0
        confs = preds[:, 4]
        best = int(confs.argmax())
        return int(preds[best, 5])

class ClassificationNode(Node):
    def __init__(self):
        super().__init__('classification_node')
        self.bridge = CvBridge()
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = YoloDetector('yolov5s_best.pt', dev)
        # subscribe to raw camera images
        self.create_subscription(Image, '/simulated_camera/image_raw', self.image_callback, 10)
        # publish detected class as Int16
        self.class_pub = self.create_publisher(Int16, '/class', 10)
        self.get_logger().info('Classification node started')
        logger.info('Classification node initialized')

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            logger.error(f"CVBridge error: {e}")
            return
        cls = self.detector.predict(cv_img)
        self.class_pub.publish(Int16(data=cls))
        self.get_logger().info(f"Published class: {cls}")
        logger.info(f"Published class: {cls}")


def main(args=None):
    rclpy.init(args=args)
    node = ClassificationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info('Shutting down classification node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()