# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
# from sensor_msgs.msg import CompressedImage, LaserScan
# from std_msgs.msg import Int16, ColorRGBA
# from visualization_msgs.msg import Marker
# from cv_bridge import CvBridge
# import numpy as np
# import math
# import torch


# def yaw_to_quaternion(yaw: float) -> Quaternion:
#     """Convert yaw (radians) to a ROS2 Quaternion message."""
#     return Quaternion(x=0.0, y=0.0,
#                       z=math.sin(yaw / 2),
#                       w=math.cos(yaw / 2))


# def quaternion_to_yaw(q: Quaternion) -> float:
#     """Convert a ROS2 Quaternion message to a yaw angle (radians)."""
#     return 2 * math.atan2(q.z, q.w)


# def calculate_distance(p: Point, target: tuple) -> float:
#     """Euclidean distance between current position and a 2D waypoint."""
#     dx = p.x - target[0]
#     dy = p.y - target[1]
#     return math.hypot(dx, dy)


# def generate_grid(top_left, top_right, bottom_left, bottom_right, rows=3, cols=6):
#     """Generate a rows x cols grid between four corner points."""
#     grid = np.zeros((rows, cols, 2))
#     for i in range(rows):
#         left = np.linspace(top_left, bottom_left, rows)[i]
#         right = np.linspace(top_right, bottom_right, rows)[i]
#         grid[i, :, :] = np.linspace(left, right, cols)
#     return grid

# # Precomputed grid points (replace with your maze corners or load dynamically)
# GRID = generate_grid(
#     top_left=(0.138, 2.78),
#     top_right=(0.0, -1.85),
#     bottom_left=(-1.65, 2.81),
#     bottom_right=(-1.85, -1.76)
# )


# class YoloDetector:
#     """Wrapper around a YOLOv5 custom model."""
#     def __init__(self, model_path: str, device: str = 'cpu'):
#         self.model = torch.hub.load(
#             'ultralytics/yolov5', 'custom', path=model_path, force_reload=False
#         ).to(device)
#         self.model.eval()
#         # adjust detection thresholds if desired
#         self.model.conf = 0.2
#         self.model.iou  = 0.45

#     def predict(self, image: np.ndarray) -> int:
#         # Expecting BGR image; convert to RGB
#         rgb_img = image[..., ::-1]
#         results = self.model(rgb_img)
#         preds = results.xyxy[0]  # tensor of shape (N,6): [x1,y1,x2,y2,conf,cls]
#         if preds.shape[0] == 0:
#             # No detection → interpret as "empty wall"
#             return 0
#         # pick detection with highest confidence
#         confs = preds[:, 4]
#         best_idx = int(confs.argmax())
#         cls = int(preds[best_idx, 5])
#         return cls


# class WaypointPublisherNode(Node):
#     def __init__(self):
#         super().__init__('waypoint_publisher_node')
#         self.bridge = CvBridge()

#         # Publishers
#         self.waypoint_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
#         self.marker_pub   = self.create_publisher(Marker, '/visualization_marker', 10)

#         # Subscribers
#         self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
#         self.create_subscription(Int16, '/class', self.classification_callback, 10)

#         # Control timer
#         self.create_timer(1.0, self.control_loop)

#         # State
#         self.current_pose = None
#         self.yaw          = 0.0
#         self.front_lidar  = None
#         self.curr_img     = None
#         self.curr_class   = None
#         self.initialized  = False
#         self.grid_x = 0
#         self.grid_y = 0
#         self.current_orient = 0
#         self.orientations = {0: math.pi/2, 1: 0, 2: -math.pi/2, 3: -math.pi}

#         # Detector
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.detector = YoloDetector('yolov5s_best.pt', device)

#     def odom_callback(self, msg):
#         pos = msg.pose.pose.position
#         ori = msg.pose.pose.orientation
#         self.current_pose = pos
#         self.yaw = quaternion_to_yaw(ori)

#         if not self.initialized:
#             # find closest grid cell index
#             flat = GRID.reshape(-1, 2)
#             dists = [calculate_distance(pos, tuple(pt)) for pt in flat]
#             idx = int(np.argmin(dists))
#             self.grid_y, self.grid_x = divmod(idx, GRID.shape[1])
#             # find closest orientation
#             self.current_orient = max(
#                 self.orientations.items(),
#                 key=lambda iv: math.cos(self.yaw - iv[1])
#             )[0]
#             self.initialized = True
#             self.publish_points()

#     def scan_callback(self, msg):
#         ranges = np.array(msg.ranges)
#         ranges = ranges[~np.isnan(ranges)]
#         self.front_lidar = float((ranges[:10].mean() + ranges[-10:].mean()) / 2)

#     def image_callback(self, msg):
#         self.curr_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')

#     def classification_callback(self, msg):
#         self.curr_class = msg.data

#     def publish_points(self):
#         marker = Marker()
#         marker.header.frame_id = 'map'
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.type = Marker.POINTS
#         marker.scale.x = marker.scale.y = 0.1
#         marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
#         for row in GRID:
#             for x, y in row:
#                 p = Point(x=x, y=y, z=0.0)
#                 marker.points.append(p)
#         self.marker_pub.publish(marker)

#     def control_loop(self):
#         if not self.initialized or self.curr_img is None or self.front_lidar is None:
#             return

#         # inch/spin to face the wall
#         cmd = Twist()
#         cmd.angular.z = 0.1
#         self.create_publisher(Twist, '/cmd_vel', 10).publish(cmd)

#         # detect sign via YOLO or override with external class
#         cls = self.detector.predict(self.curr_img)
#         if self.curr_class is not None:
#             cls = self.curr_class

#         # compute next grid coords & orientation
#         xn, yn, orient = self.compute_next(cls)
#         self.publish_waypoint(xn, yn, orient)

#     def compute_next(self, cls):
#         on = self.current_orient
#         # if forward free, stay in same orientation
#         if self.front_lidar > 0.9:
#             return self.grid_x, self.grid_y, on
#         # otherwise rotate based on sign
#         if cls == 1:      # LEFT
#             on = (on - 1) % 4
#         elif cls == 2:    # RIGHT
#             on = (on + 1) % 4
#         elif cls in (3, 4): # REVERSE or STOP
#             on = (on + 2) % 4
#         elif cls == 5:    # GOAL
#             rclpy.shutdown()
#         return self.grid_x, self.grid_y, on

#     def publish_waypoint(self, gx, gy, orient_idx):
#         wp = GRID[gy, gx]
#         msg = PoseStamped()
#         msg.header.frame_id = 'map'
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.pose.position = Point(x=wp[0], y=wp[1], z=0.0)
#         msg.pose.orientation = yaw_to_quaternion(self.orientations[orient_idx])
#         self.waypoint_pub.publish(msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = WaypointPublisherNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
# from sensor_msgs.msg import CompressedImage, LaserScan, Image
# from std_msgs.msg import Int16, ColorRGBA
# from visualization_msgs.msg import Marker
# from cv_bridge import CvBridge
# import numpy as np
# import math
# import torch

# # Constants
# ORIENTATION_TOLERANCE = 0.1  # radians
# WALL_DISTANCE_THRESHOLD = 0.9  # meters
# FORWARD_SPEED = 0.2  # m/s
# ROTATION_SPEED = 0.3  # rad/s


# def yaw_to_quaternion(yaw: float) -> Quaternion:
#     return Quaternion(x=0.0, y=0.0,
#                       z=math.sin(yaw / 2),
#                       w=math.cos(yaw / 2))


# def quaternion_to_yaw(q: Quaternion) -> float:
#     return 2 * math.atan2(q.z, q.w)


# def calculate_distance(p: Point, target: tuple) -> float:
#     dx = p.x - target[0]
#     dy = p.y - target[1]
#     return math.hypot(dx, dy)


# def generate_grid(top_left, top_right, bottom_left, bottom_right, rows=3, cols=6):
#     grid = np.zeros((rows, cols, 2))
#     for i in range(rows):
#         left = np.linspace(top_left, bottom_left, rows)[i]
#         right = np.linspace(top_right, bottom_right, rows)[i]
#         grid[i, :, :] = np.linspace(left, right, cols)
#     return grid

# # Precomputed grid points
# # GRID = generate_grid(
# #     top_left=(0.138, 2.78),
# #     top_right=(0.0, -1.85),
# #     bottom_left=(-1.65, 2.81),
# #     bottom_right=(-1.85, -1.76)
# # )

# # grid corners for gazebo
# GRID = generate_grid(
#     top_left = (4.547409534454346, 2.1913464069366455),
#     top_right = (4.5883917808532715, -0.4969266653060913),
#     bottom_left = (-0.582728028297424, 2.248652219772339),
#     bottom_right = (-0.5827378034591675, -0.4162125289440155)
# )

# class YoloDetector:
#     def __init__(self, model_path: str, device: str = 'cpu'):
#         self.model = torch.hub.load(
#             'ultralytics/yolov5', 'custom', path=model_path, force_reload=False
#         ).to(device)
#         self.model.eval()
#         self.model.conf = 0.2
#         self.model.iou  = 0.45

#     def predict(self, image: np.ndarray) -> int:
#         rgb_img = image[..., ::-1]
#         results = self.model(rgb_img)
#         preds = results.xyxy[0]
#         if preds.shape[0] == 0:
#             return 0
#         confs = preds[:, 4]
#         best_idx = int(confs.argmax())
#         return int(preds[best_idx, 5])

# class WaypointPublisherNode(Node):
#     def __init__(self):
#         super().__init__('waypoint_publisher_node')
#         self.bridge = CvBridge()

#         # Publishers
#         self.waypoint_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
#         self.marker_pub   = self.create_publisher(Marker, '/visualization_marker', 10)
#         self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel', 10)

#         # Subscribers
#         self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, 10)
#         self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
#         # comment out on gazebo testing
#         # self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
#         self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
#         self.create_subscription(Int16, '/class', self.classification_callback, 10)

#         # Control timer
#         self.create_timer(0.1, self.control_loop)

#         # State
#         self.current_pose = None
#         self.yaw          = 0.0
#         self.front_lidar  = None
#         self.curr_img     = None
#         self.curr_class   = None
#         self.initialized  = False
#         self.grid_x = 0
#         self.grid_y = 0
#         self.current_orient = 0
#         self.orientations = {0: math.pi/2, 1: 0, 2: -math.pi/2, 3: math.pi}

#         # Detector
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.detector = YoloDetector('yolov5s_best.pt', device)

#     def odom_callback(self, msg):
#         pos = msg.pose.pose.position
#         ori = msg.pose.pose.orientation
#         self.current_pose = pos
#         self.yaw = quaternion_to_yaw(ori)

#         if not self.initialized:
#             flat = GRID.reshape(-1, 2)
#             dists = [calculate_distance(pos, tuple(pt)) for pt in flat]
#             idx = int(np.argmin(dists))
#             self.grid_y, self.grid_x = divmod(idx, GRID.shape[1])
#             self.current_orient = max(
#                 self.orientations.items(),
#                 key=lambda iv: math.cos(self.yaw - iv[1])
#             )[0]
#             self.initialized = True
#             self.publish_points()

#     def scan_callback(self, msg):
#         ranges = np.array(msg.ranges)
#         ranges = ranges[~np.isnan(ranges)]
#         self.front_lidar = float((ranges[:10].mean() + ranges[-10:].mean()) / 2)

#     def image_callback(self, msg):
#         # self.curr_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
#         try:
#             # Convert a ROS2 Image msg → OpenCV BGR array
#             cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f"CVBridge error: {e}")
#             return
        
#         self.curr_img = cv_img

#     def classification_callback(self, msg):
#         self.curr_class = msg.data

#     def publish_points(self):
#         marker = Marker()
#         marker.header.frame_id = 'map'
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.type = Marker.POINTS
#         marker.scale.x = marker.scale.y = 0.1
#         marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
#         for row in GRID:
#             for x, y in row:
#                 p = Point(x=x, y=y, z=0.0)
#                 marker.points.append(p)
#         self.marker_pub.publish(marker)

#     def compute_next(self, cls):
#         on = self.current_orient
#         # update orientation based on sign
#         if cls == 1:      # LEFT
#             on = (on - 1) % 4
#         elif cls == 2:    # RIGHT
#             on = (on + 1) % 4
#         elif cls in (3, 4): # REVERSE or STOP
#             on = (on + 2) % 4
#         elif cls == 5:    # GOAL
#             return self.grid_x, self.grid_y, on
#         # compute next cell in direction 'on'
#         dx, dy = 0, 0
#         if on == 0:  dy = -1
#         elif on == 1: dx = 1
#         elif on == 2: dy = 1
#         elif on == 3: dx = -1
#         nx = max(0, min(GRID.shape[1]-1, self.grid_x + dx))
#         ny = max(0, min(GRID.shape[0]-1, self.grid_y + dy))
#         return nx, ny, on

#     def publish_waypoint(self, gx, gy, orient_idx):
#         wp = GRID[gy, gx]
#         msg = PoseStamped()
#         msg.header.frame_id = 'map'
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.pose.position = Point(x=wp[0], y=wp[1], z=0.0)
#         msg.pose.orientation = yaw_to_quaternion(self.orientations[orient_idx])
#         self.waypoint_pub.publish(msg)

#     def control_loop(self):
#         if not self.initialized or self.curr_img is None or self.front_lidar is None:
#             return

#         # determine target orientation quaternion
#         target_yaw = self.orientations[self.current_orient]
#         yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))

#         cmd = Twist()
#         # rotate first to align
#         if abs(yaw_error) > ORIENTATION_TOLERANCE:
#             cmd.angular.z = ROTATION_SPEED * (1 if yaw_error > 0 else -1)
#         else:
#             # facing wall; decide whether to move or classify
#             if self.front_lidar > WALL_DISTANCE_THRESHOLD:
#                 cmd.linear.x = FORWARD_SPEED
#             else:
#                 # classify sign when stationary
#                 cls = self.detector.predict(self.curr_img)
#                 if self.curr_class is not None:
#                     cls = self.curr_class
#                 nx, ny, orient = self.compute_next(cls)
#                 if cls == 5:
#                     rclpy.shutdown()
#                     return
#                 # update internal state
#                 self.grid_x, self.grid_y, self.current_orient = nx, ny, orient
#                 self.publish_waypoint(nx, ny, orient)
#                 # reset override classification
#                 self.curr_class = None
#         # publish movement command
#         self.cmd_pub.publish(cmd)


# def main(args=None):
#     rclpy.init(args=args)
#     node = WaypointPublisherNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import logging
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Int16, ColorRGBA
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from collections import deque
import numpy as np
import math
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)

# Configure Python logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ORIENTATION_TOLERANCE = 0.1  # radians
WALL_DISTANCE_THRESHOLD = 0.4  # meters
FORWARD_SPEED = 0.13  # m/s
ROTATION_SPEED = 0.1  # rad/s

amcl_qos = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE
)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0,
                      z=math.sin(yaw / 2),
                      w=math.cos(yaw / 2))


def quaternion_to_yaw(q: Quaternion) -> float:
    return 2 * math.atan2(q.z, q.w)


def calculate_distance(p: Point, target: tuple) -> float:
    return math.hypot(p.x - target[0], p.y - target[1])


def generate_grid(top_left, top_right, bottom_left, bottom_right, rows=3, cols=6):
    grid = np.zeros((rows, cols, 2))
    for i in range(rows):
        left = np.linspace(top_left, bottom_left, rows)[i]
        right = np.linspace(top_right, bottom_right, rows)[i]
        grid[i, :, :] = np.linspace(left, right, cols)
    return grid

# Precomputed grid points for gazebo
# GRID = generate_grid(
#     top_left=(4.5474, 2.1913),
#     top_right=(4.5884, -0.4969),
#     bottom_left=(-0.5827, 2.2487),
#     bottom_right=(-0.5827, -0.4162)
# )

GRID = generate_grid(
    top_left=(-0.5827, 2.2487),
    top_right=(4.5474, 2.1913),
    bottom_left=(-0.5827, -0.4162),
    bottom_right=(4.5884, -0.4969),
)


class YoloDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        logger.info(f"Loading YOLO model from {model_path} on {device}")
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=model_path, force_reload=True
        ).to(device)
        self.model.eval()
        self.model.conf = 0.2
        self.model.iou = 0.45


    def predict(self, image: np.ndarray) -> int:
        rgb = image[..., ::-1]
        results = self.model(rgb)
        preds = results.xyxy[0]
        if preds.shape[0] == 0:
            return 0
        confs = preds[:, 4]
        best = int(confs.argmax())
        return int(preds[best, 5])


class WaypointPublisherNode(Node):
    def __init__(self):
        super().__init__('waypoint_publisher_node')
        self.get_logger().info("Initializing node")
        self.bridge = CvBridge()
        self.state = 'MOVING'
        # buffer last 5 predictions to filter out spurious reads
        self.pred_buffer = deque(maxlen=5)

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Subscribers
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, amcl_qos)
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,  # Keeps the last N messages
            depth=1,  # The depth of the message queue
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Ensures reliable message delivery
        )
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.create_subscription(Image, '/simulated_camera/image_raw', self.image_callback, 10)
        self.create_subscription(Int16, '/class', self.classification_callback, QoSProfile(depth=10))

        # Timer
        self.create_timer(0.1, self.control_loop)

        # State
        self.current_pose = None
        self.yaw = 0.0
        self.front_lidar = None
        self.curr_img = None
        self.curr_class = None
        self.initialized = False
        self.grid_x = 0
        self.grid_y = 0
        self.current_orient = 0
        self.orientations = {0: math.pi/2, 1: 0, 2: -math.pi/2, 3: math.pi}

        # Detector
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = YoloDetector('yolov5s_best.pt', dev)


    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.current_pose = pos
        self.yaw = quaternion_to_yaw(ori)
        if not self.initialized:
            flat = GRID.reshape(-1, 2)
            dists = [calculate_distance(pos, tuple(p)) for p in flat]
            idx = int(np.argmin(dists))
            self.grid_y, self.grid_x = divmod(idx, GRID.shape[1])
            self.current_orient = max(
                self.orientations.items(),
                key=lambda x: math.cos(self.yaw - x[1])
            )[0]
            self.initialized = True
            self.get_logger().info(f"Init done: grid=({self.grid_x},{self.grid_y}) orient={self.current_orient}")
            self.publish_points()


    def scan_callback(self, msg):
        arr = np.array(msg.ranges)
        arr = arr[~np.isnan(arr)]
        self.front_lidar = float((arr[:10].mean() + arr[-10:].mean()) / 2)


    def image_callback(self, msg):
        try:
            self.curr_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge: {e}")


    def classification_callback(self, msg):
        self.curr_class = msg.data


    def publish_points(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.scale.x = marker.scale.y = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        # Add all grid points to the marker
        for row in GRID:
            for x, y in row:
                p = Point(x=x, y=y, z=0.0)
                marker.points.append(p)

        # Highlight the current position on the grid
        current_marker = Marker()
        current_marker.header.frame_id = 'map'
        current_marker.header.stamp = self.get_clock().now().to_msg()
        current_marker.type = Marker.SPHERE
        current_marker.scale.x = current_marker.scale.y = current_marker.scale.z = 0.2
        current_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        current_marker.pose.position = Point(
            x=GRID[self.grid_y, self.grid_x, 0],
            y=GRID[self.grid_y, self.grid_x, 1],
            z=0.0
        )

        # Publish both markers
        self.marker_pub.publish(marker)
        self.marker_pub.publish(current_marker)


    def compute_next(self, cls):
        o = self.current_orient
        if cls == 1:
            logger.info("Turn Left")
            o = (o-1) % 4
        elif cls == 2:
            logger.info("Turn Right")
            o = (o+1) % 4
        elif cls in (3, 4):
            logger.info("Don't enter, Stop")
            o = (o+2) % 4
        elif cls == 5:
            logger.info("Fuck yea")
            return self.grid_x, self.grid_y, o
        else:
            logger.info("No sign → moving straight (cls=0)")
            return self.grid_x, self.grid_y, o

        dx, dy = 0, 0
        if o == 0: dy = -1
        elif o == 1: dx = 1
        elif o == 2: dy = 1
        elif o == 3: dx = -1

        nx = max(0, min(self.grid_x + dx, GRID.shape[1] - 1))
        ny = max(0, min(self.grid_y + dy, GRID.shape[0] - 1))
        return nx, ny, o


    def publish_waypoint(self, gx, gy, orient_idx):
        wp = GRID[gy, gx]
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position = Point(x=wp[0], y=wp[1], z=0.0)
        msg.pose.orientation = yaw_to_quaternion(self.orientations[orient_idx])
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"Waypoint → ({wp[0]:.3f},{wp[1]:.3f}) orient={orient_idx}")


    def control_loop(self):
        # Calculate yaw error relative to the wall direction
        target_yaw = self.orientations[self.current_orient]
        yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
        # Rotate in place until aligned within tolerance
        if abs(yaw_error) > ORIENTATION_TOLERANCE and self.state == 'MOVING':
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = ROTATION_SPEED * (1 if yaw_error > 0 else -1)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"Aligning: yaw_error={yaw_error:.3f} → angular.z={cmd.angular.z:.3f}"
            )
            return

        # Ensure initialization and sensor data before moving or classifying
        if not (self.initialized and self.curr_img is not None and self.front_lidar is not None):
            return

        # FSM
        if self.state == 'MOVING':
            if self.front_lidar > WALL_DISTANCE_THRESHOLD:
                # Move forward if there's space
                cmd = Twist()
                cmd.linear.x = FORWARD_SPEED
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"Moving forward: lidar={self.front_lidar:.3f}")
                return
            else:
                self.state = 'CLASSIFY'
                self.pred_buffer.clear()
                self.get_logger().info("Wall detected; switching to CLASSIFY state")
                return

        if self.state == 'CLASSIFY':
            cls = self.curr_class if self.curr_class is not None else self.detector.predict(self.curr_img)
            self.pred_buffer.append(cls)

            # Check if we have enough predictions to make a decision
            if len(self.pred_buffer) == self.pred_buffer.maxlen:
                unique_preds = set(self.pred_buffer)
                if len(unique_preds) == 1 and 0 in unique_preds:
                    cmd = Twist()
                    cmd.angular.z = ROTATION_SPEED
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info("Empty wall; rotating to find sign")
                    # self.pred_buffer.clear()
                    return
                elif len(unique_preds) == 1:
                    cls = unique_preds.pop()
                    self.get_logger().info(f"Classified sign: {cls}")
                    # Compute next waypoint based on classification
                    nx, ny, new_o = self.compute_next(cls)
                    if cls == 5:
                        logger.info("Goal sign detected; shutting down node.")
                        rclpy.shutdown()
                        return
                    # Update grid position & orientation
                    self.grid_x, self.grid_y, self.current_orient = nx, ny, new_o
                    self.publish_waypoint(nx, ny, new_o)
                    self.get_logger().info(f"Published next waypoint: grid=({nx},{ny}), orient={new_o}")
                    self.curr_class = None
                    self.state = 'MOVING'
                    self.get_logger().info(f"Committed cls={cls}; → STATE = MOVING")
                    return
            return
        return

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    logger.info("Starting ROS2 node")
    node = WaypointPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

