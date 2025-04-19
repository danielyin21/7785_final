#!/usr/bin/env python3
import rclpy, math, time, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist

# Sign IDs (must match your classifier)
LEFT, RIGHT, DNE, STOP, GOAL = 1, 2, 3, 4, 5
VALID_SIGNS = {LEFT, RIGHT, DNE, STOP, GOAL}

class PD:
    """Simple PD controller: err → angular velocity."""
    def __init__(self, kp, kd, out_lim=1.2):
        self.kp, self.kd, self.out_lim = kp, kd, out_lim
        self.prev_e = 0.0
        self.prev_t = None

    def reset(self):
        self.prev_e = 0.0
        self.prev_t = None

    def step(self, err):
        now = time.time()
        if self.prev_t is None:
            self.prev_t, self.prev_e = now, err
            return 0.0
        dt = now - self.prev_t
        de = err - self.prev_e
        self.prev_t, self.prev_e = now, err
        out = self.kp * err + self.kd * (de / dt)
        return max(-self.out_lim, min(self.out_lim, out))

class CombinedNavigator(Node):
    def __init__(self):
        super().__init__('combined_navigator')

        # Parameters (can be overridden via ROS2 params)
        self.declare_parameter('forward_speed', 0.15)
        self.declare_parameter('stop_distance', 0.50)
        self.declare_parameter('scan_fov_deg', 60.0)
        self.declare_parameter('kp_lin', 4.0)
        self.declare_parameter('kd_lin', 0.5)
        self.declare_parameter('kp_rot', 3.3)
        self.declare_parameter('kd_rot', 0.4)
        self.declare_parameter('shake_angle_deg', 30.0)

        # Drive-to-wall params
        self.speed    = self.get_parameter('forward_speed').value
        self.dstop    = self.get_parameter('stop_distance').value
        self.fov_rad  = math.radians(self.get_parameter('scan_fov_deg').value)
        kp_lin        = self.get_parameter('kp_lin').value
        kd_lin        = self.get_parameter('kd_lin').value
        self.drive_pd = PD(kp_lin, kd_lin)

        # Rotation params
        kp_rot        = self.get_parameter('kp_rot').value
        kd_rot        = self.get_parameter('kd_rot').value
        self.rotate_pd = PD(kp_rot, kd_rot)
        self.shake_deg = self.get_parameter('shake_angle_deg').value

        # State
        self.yaw = 0.0
        self.yaw_ref_drive = None
        self.arrived = False

        self.rotation_state = 'WAIT_WALL'
        self.yaw_orig = None
        self.target_yaw = None
        self.sign = None

        # Publishers & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan,  '/scan', self.scan_cb, 10)
        # self.create_subscription(Odometry,   '/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, 10)
        self.create_subscription(Int16,      '/class', self.class_cb, 10)

        # Main loop
        self.create_timer(0.05, self.tick)

        self.get_logger().info('CombinedNavigator started')

    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

    def scan_cb(self, scan: LaserScan):
        # compute min distance in front FOV
        half = self.fov_rad/2.0
        start = int((-half - scan.angle_min)/scan.angle_increment)
        end   = int(( half - scan.angle_min)/scan.angle_increment)
        start = max(0, start); end = min(len(scan.ranges)-1, end)
        seg = np.array(scan.ranges[start:end+1])
        dist = float(np.nanmin(seg)) if seg.size else float('inf')

        if not self.arrived and dist < self.dstop:
            self.arrived = True
            self.yaw_orig = self.yaw
            self.rotation_state = 'DECIDE'
            self.rotate_pd.reset()
            self.get_logger().info('Arrived at wall, switching to rotation')

    def class_cb(self, msg: Int16):
        sid = int(msg.data)
        if sid in VALID_SIGNS:
            self.sign = sid
            # if shaking, interrupt back-to-original to rotate sign
            if self.rotation_state in ('SHAKE_LEFT', 'SHAKE_RIGHT'):
                self.rotation_state = 'SHAKE_BACK'
                self.rotate_pd.reset()

    def tick(self):
        if not self.arrived:
            self._drive_to_wall()
        else:
            self._rotate_to_sign()

    def _drive_to_wall(self):
        tw = Twist()
        # initialize reference yaw once
        if self.yaw_ref_drive is None:
            self.yaw_ref_drive = self.yaw
            self.drive_pd.reset()
        err = (self.yaw_ref_drive - self.yaw + math.pi) % (2*math.pi) - math.pi
        tw.angular.z = self.drive_pd.step(err)
        tw.linear.x  = self.speed
        self.cmd_pub.publish(tw)

    def _rotate_to_sign(self):
        st = self.rotation_state
        if st == 'DECIDE':
            if self.sign is not None:
                self._prep_sign_rotation()
            else:
                self.target_yaw = (self.yaw_orig + math.radians(self.shake_deg))%(2*math.pi)
                self.rotation_state = 'SHAKE_LEFT'
                self.rotate_pd.reset()

        elif st == 'SHAKE_LEFT':
            if self._reach_target():
                self.target_yaw = (self.yaw_orig - math.radians(self.shake_deg))%(2*math.pi)
                self.rotation_state = 'SHAKE_RIGHT'
                self.rotate_pd.reset()

        elif st == 'SHAKE_RIGHT':
            if self._reach_target():
                self.target_yaw = self.yaw_orig
                self.rotation_state = 'SHAKE_BACK'
                self.rotate_pd.reset()

        elif st == 'SHAKE_BACK':
            if self._reach_target():
                if self.sign is not None:
                    self._prep_sign_rotation()
                else:
                    self.rotation_state = 'WAIT_WALL'
                    self.get_logger().info('No sign detected, ready for next wall')

        elif st == 'SIGN_ROTATE':
            if self._reach_target():
                self.rotation_state = 'WAIT_WALL'
                self.arrived        = False       # allow driving again
                self.yaw_ref_drive  = None        # reset forward‐drive reference
                self.sign           = None        # clear previous sign
                self.get_logger().info('Sign rotation complete')

        # WAIT_WALL: do nothing

    def _prep_sign_rotation(self):
        self.yaw_orig = self.yaw
        if   self.sign == LEFT:  offset =  math.pi/2
        elif self.sign == RIGHT: offset = -math.pi/2
        elif self.sign in (DNE, STOP): offset = math.pi
        elif self.sign == GOAL:
            self.get_logger().info('GOAL reached! stopping.')
            self.cmd_pub.publish(Twist())
            self.rotation_state = 'WAIT_WALL'
            return
        else:
            offset = 0.0
        self.target_yaw = (self.yaw_orig + offset)%(2*math.pi)
        self.rotation_state = 'SIGN_ROTATE'
        self.rotate_pd.reset()

    def _ang_err(self, tgt):
        return (tgt - self.yaw + math.pi)%(2*math.pi) - math.pi

    def _reach_target(self):
        err = self._ang_err(self.target_yaw)
        if abs(err) < math.radians(0.5):
            self.cmd_pub.publish(Twist())
            return True
        omega = self.rotate_pd.step(err)
        # enforce minimum turn
        if 0 < abs(omega) < 0.15:
            omega = 0.15 * math.copysign(1, omega)
        tw = Twist(); tw.angular.z = omega
        self.cmd_pub.publish(tw)
        return False

def main(args=None):
    rclpy.init(args=args)
    node = CombinedNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
