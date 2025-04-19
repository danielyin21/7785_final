#!/usr/bin/env python3
import rclpy, torch, cv2
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from torchvision import transforms
from PIL import Image

class SignClassifier(Node):
    def __init__(self):
        super().__init__('sign_classifier')
        self.get_logger().info('Sign Classifier Node Started')

        self.br = CvBridge()

        self.model = torch.load('2025G_model2.pkl', map_location=torch.device('cpu'), weights_only=False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.cb,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(Int32, '/sign_id', 10)

    def cb(self, msg):
        try:
            img = self.br.compressed_imgmsg_to_cv2(msg, 'bgr8')

            cv2.imshow("TurtleBot3 Camera", img)
            cv2.waitKey(1)

            input_tensor = self.preprocess(img)

            with torch.no_grad():
                pred = self.model(input_tensor).argmax().item()

            self.pub.publish(Int32(data=pred))
            self.get_logger().info(f"Predicted Sign ID: {pred}")

        except Exception as e:
            self.get_logger().error(f"Prediction failed: {str(e)}")

    def preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0)
        return input_tensor.to(torch.float)

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
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
