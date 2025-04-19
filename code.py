### drive_to_wall.py

#!/usr/bin/env python3

"""

DriveToWall (PD 版)

-------------------

* 当前使用 PD 控制器保持航向：角速度 = kp * 误差 + kd * d(误差)/dt

* 当前未使用积分项，防止航向飘移。

* 到达墙面距离内停止并发布 /arrive_wall。

"""

import rclpy

import numpy as np

import math, time

from rclpy.node import Node

from sensor_msgs.msg import LaserScan

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

from std_msgs.msg import Bool

from rclpy.qos import qos_profile_sensor_data
 
class PD:

    def __init__(self, kp, kd, out_lim=1.2):

        self.kp, self.kd = kp, kd

        self.out_lim = out_lim

        self.prev_e = 0.0

        self.prev_t = None

    def reset(self):

        self.prev_e = 0.0; self.prev_t = None

    def step(self, err):

        now = time.time()

        if self.prev_t is None:

            self.prev_t = now

            self.prev_e = err

            return 0.0

        dt = now - self.prev_t

        de = err - self.prev_e

        self.prev_t = now; self.prev_e = err

        omega = self.kp * err + self.kd * (de / dt)

        return max(-self.out_lim, min(self.out_lim, omega))
 
class DriveToWall(Node):

    def __init__(self):

        super().__init__('drive_to_wall')
 
        # 参数

        self.declare_parameter('forward_speed', 0.15)

        self.declare_parameter('stop_distance', 0.50)

        self.declare_parameter('scan_fov_deg', 60.0)

        self.declare_parameter('kp_lin', 4.0)

        self.declare_parameter('kd_lin', 0.5)
 
        self.speed   = self.get_parameter('forward_speed').value

        self.dstop   = self.get_parameter('stop_distance').value

        self.fov_rad = math.radians(self.get_parameter('scan_fov_deg').value)

        kp = self.get_parameter('kp_lin').value

        kd = self.get_parameter('kd_lin').value
 
        # 控制器

        self.pd = PD(kp, kd)
 
        # 状态

        self.moving    = True

        self.rotating  = False

        self.yaw       = 0.0

        self.yaw_ref   = None
 
        # 通信

        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)

        self.arrive_pub = self.create_publisher(Bool,  '/arrive_wall', 10)

        self.create_subscription(LaserScan, '/scan',  self.scan_cb,  qos_profile_sensor_data)

        self.create_subscription(Odometry,  '/odom',  self.odom_cb, 10)

        self.create_subscription(Bool,      '/rotating', self.rotating_cb, 10)

        self.create_timer(0.05, self.drive)
 
    def odom_cb(self, msg: Odometry):

        q = msg.pose.pose.orientation

        siny = 2.0 * (q.w*q.z + q.x*q.y)

        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)

        self.yaw = math.atan2(siny, cosy)
 
    def rotating_cb(self, msg: Bool):

        self.rotating = msg.data

        if self.rotating:

            self.cmd_pub.publish(Twist())

            self.moving  = False

            self.yaw_ref = None

        else:

            self.yaw_ref = self.yaw

            self.pd.reset()
 
    def scan_cb(self, scan: LaserScan):

        if self.rotating:

            return

        half  = self.fov_rad / 2.0

        start = int((-half - scan.angle_min) / scan.angle_increment)

        end   = int(( half - scan.angle_min) / scan.angle_increment)

        start = max(0, start); end = min(len(scan.ranges)-1, end)

        seg   = np.array(scan.ranges[start:end+1])

        dist  = float(np.nanmin(seg)) if seg.size else float('inf')
 
        if dist < self.dstop:

            if self.moving:

                self.cmd_pub.publish(Twist())

                self.arrive_pub.publish(Bool(data=True))

            self.moving = False

            self.arrive_pub.publish(Bool(data=True))

        else:

            self.moving = True

            if self.yaw_ref is None:

                self.yaw_ref = self.yaw

                self.pd.reset()
 
    def drive(self):

        twist = Twist()

        if self.moving and not self.rotating:

            if self.yaw_ref is not None:

                err = (self.yaw_ref - self.yaw + math.pi) % (2*math.pi) - math.pi

                twist.angular.z = self.pd.step(err)

            twist.linear.x = self.speed

        self.cmd_pub.publish(twist)
 
# main
 
def main():

    rclpy.init()

    node = DriveToWall()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()
 
if __name__ == '__main__':

    main()
 
### rotate_navigator.py

#!/usr/bin/env python3

"""

RotateNavigator – 摇头找标志 + **PD** 角度控制

------------------------------------------------

* 状态机不变：WAIT_WALL → DECIDE → SHAKE_LEFT → SHAKE_RIGHT → SHAKE_BACK → SIGN_ROTATE。

* 控制器改为 **PD**（无积分项）：

    ω = kp · err + kd · d(err)/dt

* 仍保证：连续相同方向标志会正确旋转；先回到原角再按标志偏转。

"""

import rclpy, math, time

from rclpy.node import Node

from std_msgs.msg import Bool, Int32

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry
 
LEFT, RIGHT, DNE, STOP, GOAL = 1, 2, 3, 4, 5

VALID_SIGNS = {LEFT, RIGHT, DNE, STOP, GOAL}
 
class PD:

    """简易 PD 控制器 (rad → rad/s)"""

    def __init__(self, kp, kd, out_lim=1.2):

        self.kp, self.kd = kp, kd

        self.out_lim = out_lim

        self.prev_e  = 0.0

        self.prev_t  = None

    def reset(self):

        self.prev_e = 0.0; self.prev_t = None

    def step(self, err):

        now = time.time()

        if self.prev_t is None:

            self.prev_t = now

            self.prev_e = err

            return 0.0

        dt = now - self.prev_t

        de = err - self.prev_e

        self.prev_t = now; self.prev_e = err

        out = self.kp * err + self.kd * (de/dt)

        return max(-self.out_lim, min(self.out_lim, out))
 
class RotateNavigator(Node):

    def __init__(self):

        super().__init__('rotate_navigator')

        self.state = 'WAIT_WALL'

        self.sign  = None

        self.yaw       = 0.0

        self.yaw_orig  = None

        self.target_yaw= None

        self.shake_deg = 30.0

        # PD 参数

        kp, kd = 3.3, 0.4

        self.ctrl = PD(kp, kd)

        # 通信

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.rot_pub = self.create_publisher(Bool,  '/rotating', 10)

        self.create_subscription(Bool,    '/arrive_wall', self.wall_cb, 10)

        self.create_subscription(Int32,   '/sign_id',     self.sign_cb, 10)

        self.create_subscription(Odometry,'/odom',        self.odom_cb, 10)

        self.create_timer(0.02, self.tick)
 
    # ----- 回调 -----

    def odom_cb(self, msg: Odometry):

        q = msg.pose.pose.orientation

        siny = 2*(q.w*q.z + q.x*q.y)

        cosy = 1 - 2*(q.y*q.y + q.z*q.z)

        self.yaw = math.atan2(siny, cosy)
 
    def wall_cb(self, msg: Bool):

        if msg.data and self.state == 'WAIT_WALL':

            self.yaw_orig = self.yaw

            self.sign     = None

            self.state    = 'DECIDE'

            self.rot_pub.publish(Bool(data=True))
 
    def sign_cb(self, msg: Int32):

        if msg.data in VALID_SIGNS:

            self.sign = msg.data

            if self.state in ('SHAKE_LEFT', 'SHAKE_RIGHT'):

                self.state      = 'SHAKE_BACK'

                self.target_yaw = self.yaw_orig

                self.ctrl.reset()
 
    # ----- 工具 -----

    def ang_err(self, tgt):

        return (tgt - self.yaw + math.pi) % (2*math.pi) - math.pi
 
    def send_omega(self, omega):

        tw = Twist(); tw.angular.z = omega; self.cmd_pub.publish(tw)
 
    def reach_target(self) -> bool:

        err = self.ang_err(self.target_yaw)

        if abs(err) < math.radians(1.5):

            self.cmd_pub.publish(Twist())

            return True

        omega = self.ctrl.step(err)

        if 0 < abs(omega) < 0.15:

            omega = 0.15 * math.copysign(1, omega)

        self.send_omega(omega)

        return False
 
    # ----- 主循环 -----

    def tick(self):

        if self.state == 'DECIDE':

            if self.sign:

                self.prepare_sign_rotation()

            else:

                self.target_yaw = (self.yaw_orig + math.radians(self.shake_deg)) % (2*math.pi)

                self.state = 'SHAKE_LEFT'; self.ctrl.reset()

            return
 
        if self.state == 'SHAKE_LEFT':

            if self.reach_target():

                self.target_yaw = (self.yaw_orig - math.radians(self.shake_deg)) % (2*math.pi)

                self.state = 'SHAKE_RIGHT'; self.ctrl.reset()

            return
 
        if self.state == 'SHAKE_RIGHT':

            if self.reach_target():

                self.target_yaw = self.yaw_orig; self.state = 'SHAKE_BACK'; self.ctrl.reset()

            return
 
        if self.state == 'SHAKE_BACK':

            if self.reach_target():

                if self.sign:

                    self.prepare_sign_rotation()

                else:

                    self.rot_pub.publish(Bool(data=False))

                    self.state = 'WAIT_WALL'

            return
 
        if self.state == 'SIGN_ROTATE':

            if self.reach_target():

                self.rot_pub.publish(Bool(data=False))

                self.state = 'WAIT_WALL'

            return
 
    # ----- 标志处理 -----

    def prepare_sign_rotation(self):

        self.yaw_orig = self.yaw

        if   self.sign == LEFT:  offset =  math.pi/2

        elif self.sign == RIGHT: offset = -math.pi/2

        elif self.sign in (DNE, STOP): offset = math.pi

        elif self.sign == GOAL:

            self.get_logger().info('Goal reached!')

            self.cmd_pub.publish(Twist()); self.rot_pub.publish(Bool(data=False))

            self.state = 'WAIT_WALL'; return

        else:

            offset = 0.0

        self.target_yaw = (self.yaw_orig + offset) % (2*math.pi)

        self.state = 'SIGN_ROTATE'; self.ctrl.reset()
 
# ----- main -----
 
def main():

    rclpy.init(); node = RotateNavigator(); rclpy.spin(node)

    node.destroy_node(); rclpy.shutdown()
 
if __name__ == '__main__':

    main()

 