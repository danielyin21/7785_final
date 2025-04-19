# waypoint_publisher_node.py
#!/usr/bin/env python3
import logging
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16, ColorRGBA
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from collections import deque
import numpy as np
import math

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ORIENTATION_TOLERANCE = 0.1  # radians
WALL_DISTANCE_THRESHOLD = 0.4  # meters
FORWARD_SPEED = 0.13  # m/s
ROTATION_SPEED = 0.1  # rad/s

# QoS for AMCL
amcl_qos = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE
)

# grid generation

def generate_grid(top_left, top_right, bottom_left, bottom_right, rows=3, cols=6):
    grid = np.zeros((rows, cols, 2))
    for i in range(rows):
        left = np.linspace(top_left, bottom_left, rows)[i]
        right = np.linspace(top_right, bottom_right, rows)[i]
        grid[i, :, :] = np.linspace(left, right, cols)
    return grid

GRID = generate_grid(
    top_left=(-0.5827, 2.2487),
    top_right=(4.5474, 2.1913),
    bottom_left=(-0.5827, -0.4162),
    bottom_right=(4.5884, -0.4969)
)

class WaypointPublisherNode(Node):
    def __init__(self):
        super().__init__('waypoint_publisher_node')
        self.get_logger().info('Initializing robot node')
        logger.info('Robot node initialization')
        self.bridge = CvBridge()
        self.state = 'MOVING'
        self.pred_buffer = deque(maxlen=5)

        # Publishers
        self.waypoint_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.marker_pub   = self.create_publisher(Marker, '/visualization_marker', 10)
        self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, amcl_qos)
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Int16, '/class', self.classification_callback, QoSProfile(depth=10))
        self.create_timer(0.2, self.control_loop)

        # State
        self.current_pose = None
        self.yaw = 0.0
        self.front_lidar = None
        self.curr_class = None
        self.initialized = False
        self.grid_x = 0
        self.grid_y = 0
        self.current_orient = 0
        self.orientations = {0: math.pi/2, 1: 0, 2: -math.pi/2, 3: math.pi}

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.current_pose = pos
        self.yaw = 2 * math.atan2(ori.z, ori.w)
        if not self.initialized:
            flat = GRID.reshape(-1, 2)
            dists = [math.hypot(pos.x-p[0], pos.y-p[1]) for p in flat]
            idx = int(np.argmin(dists))
            self.grid_y, self.grid_x = divmod(idx, GRID.shape[1])
            self.current_orient = max(self.orientations.items(), key=lambda x: math.cos(self.yaw - x[1]))[0]
            self.initialized = True
            self.publish_points()
            self.get_logger().info(f"Init done: grid=({self.grid_x},{self.grid_y}) orient={self.current_orient}")
            logger.info(f"Init at cell ({self.grid_x},{self.grid_y}), orient {self.current_orient}")

    def scan_callback(self, msg):
        arr = np.array(msg.ranges)
        arr = arr[~np.isnan(arr)]
        self.front_lidar = float((arr[:10].mean() + arr[-10:].mean()) / 2)

    def classification_callback(self, msg):
        self.curr_class = msg.data
        self.get_logger().info(f"Received class: {self.curr_class}")
        logger.info(f"Class msg: {self.curr_class}")

    def publish_points(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.scale.x = marker.scale.y = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        for row in GRID:
            for x,y in row:
                marker.points.append(Point(x=x,y=y,z=0.0))
        current = Marker()
        current.header.frame_id = 'map'
        current.header.stamp = self.get_clock().now().to_msg()
        current.type = Marker.SPHERE
        current.scale.x = current.scale.y = current.scale.z = 0.2
        current.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        current.pose.position = Point(x=GRID[self.grid_y,self.grid_x,0], y=GRID[self.grid_y,self.grid_x,1], z=0.0)
        self.marker_pub.publish(marker)
        self.marker_pub.publish(current)
        self.get_logger().info("Published grid and current position markers")
        logger.info("Markers published to /visualization_marker")

    def compute_next(self, cls):
        o = self.current_orient
        if cls == 1:
            logger.info("Turn Left")
            o = (o-1) % 4
        elif cls == 2:
            logger.info("Turn Right")
            o = (o+1) % 4
        elif cls in (3,4):
            logger.info("Reverse/Stop")
            o = (o+2) % 4
        elif cls == 5:
            logger.info("Goal detected")
            return self.grid_x, self.grid_y, o
        else:
            logger.info("No sign; go straight")
        dx,dy = 0,0
        if o==0: dy=-1
        elif o==1: dx=1
        elif o==2: dy=1
        elif o==3: dx=-1
        nx = max(0, min(self.grid_x+dx, GRID.shape[1]-1))
        ny = max(0, min(self.grid_y+dy, GRID.shape[0]-1))
        logger.info(f"Next cell: ({nx},{ny}), orient {o}")
        return nx,ny,o

    def publish_waypoint(self, gx, gy, orient_idx):
        wp = GRID[gy,gx]
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position = Point(x=wp[0], y=wp[1], z=0.0)
        msg.pose.orientation = Quaternion(x=0.0,y=0.0,z=math.sin(self.orientations[orient_idx]/2), w=math.cos(self.orientations[orient_idx]/2))
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"Waypoint → ({wp[0]:.3f},{wp[1]:.3f}) orient={orient_idx}")
        logger.info(f"Published goal_pose ({wp[0]:.3f},{wp[1]:.3f}) orient idx {orient_idx}")

    def control_loop(self):
        target_yaw = self.orientations[self.current_orient]
        yaw_error = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
        if abs(yaw_error)>ORIENTATION_TOLERANCE and self.state=='MOVING':
            cmd = Twist()
            cmd.angular.z = ROTATION_SPEED*(1 if yaw_error>0 else -1)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(f"Aligning: yaw_err={yaw_error:.3f}, cmd.angular.z={cmd.angular.z:.3f}")
            logger.info("Rotating to align with wall")
            return
        
        if not (self.initialized and self.front_lidar is not None and self.curr_class is not None):
            return
        
        if self.state=='MOVING':
            if self.front_lidar>WALL_DISTANCE_THRESHOLD:
                cmd=Twist(); cmd.linear.x=FORWARD_SPEED
                self.cmd_pub.publish(cmd)
                self.get_logger().info(f"Moving forward: lidar={self.front_lidar:.3f}")
                logger.info("Forward cmd published")
                return
            else:
                self.state='CLASSIFY'; self.pred_buffer.clear()
                self.get_logger().info("Switched to CLASSIFY")
                logger.info("Entering classification state")
                return
        
        if self.state=='CLASSIFY':
            cls = self.curr_class
            self.pred_buffer.append(cls)
            if len(self.pred_buffer)==self.pred_buffer.maxlen:
                unique = set(self.pred_buffer)
                if len(unique)==1 and 0 in unique:
                    cmd=Twist(); cmd.angular.z=ROTATION_SPEED * 10
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info("Empty wall; rotating")
                    logger.info("Empty classification; rotating to scan")
                    self.pred_buffer.clear()
                    return
                elif len(unique)==1:
                    cls = unique.pop()
                    self.get_logger().info(f"Classified sign: {cls}")
                    logger.info(f"Final class decision: {cls}")
                    nx,ny,new_o = self.compute_next(cls)
                    if cls==5:
                        logger.info("Goal: shutting down node")
                        rclpy.shutdown()
                        return
                    self.grid_x,self.grid_y,self.current_orient = nx,ny,new_o
                    self.publish_waypoint(nx,ny,new_o)
                    self.curr_class = None
                    self.state = 'MOVING'
                    self.get_logger().info("State = MOVING")
                    logger.info("Committed waypoint and resumed MOVING state")
                    return
            return


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info('Shutting down robot node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
