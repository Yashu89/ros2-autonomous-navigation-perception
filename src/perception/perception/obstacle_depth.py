import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class DepthObstacleDetector(Node):
    def __init__(self):
        super().__init__('depth_obstacle_detector')

        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(Image, '/camera/image_raw/depth_image', self.depth_callback, 10)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/image_raw/camera_info', self.camera_info_callback, 10)

        self.obstacle_pub = self.create_publisher(PointCloud2, '/perception/depth_obstacles', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_depth = None
        self.declare_parameter('show_gui', True)
        self.show_gui = self.get_parameter('show_gui').value

        self.get_logger().info('Depth Obstacle Detector Started')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        self.latest_depth = depth

        if self.fx is None:
            return

        self.process_depth(depth, stamp=msg.header.stamp)

    def process_depth(self, depth, stamp=None):
        height, width = depth.shape

        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
        except Exception:
            return

        #processing every 4th pixel

        step = 4

        obstacle_points = []

        display = np.zeros((height, width, 3), dtype=np.uint8) if self.show_gui else None

        for v in range(0, height, step):
            for u in range(0, width, step):
                z = float(depth[v, u])

                if not np.isfinite(z):
                    continue
                if z <= 0.0:
                    continue
                if z > 4.0:
                    continue

                x_optical = ((u - self.cx) * z / self.fx)
                y_optical = ((v - self.cy) * z / self.fy) 
                z_optical = z

                x_camera = z_optical
                y_camera = -x_optical
                z_camera = -y_optical

                point = PointStamped()

                point.header.frame_id = 'camera_link'
                if stamp is not None:
                    point.header.stamp = stamp
                else:
                    point.header.stamp = self.get_clock().now().to_msg()

                point.point.x = float(x_camera)
                point.point.y = float(y_camera)
                point.point.z = float(z_camera)

                transformed = do_transform_point(
                    point,
                    transform
                )

                bx = transformed.point.x
                by = transformed.point.y
                bz = transformed.point.z

                if bx < 0.20:
                    continue
                if bx > 4.0:
                    continue
                if abs(by) > 2.0:
                    continue
                if bz < -0.07:
                    continue
                if bz > 1.5:
                    continue

                obstacle_points.append((bx, by, bz, u, v))

        if self.show_gui and display is not None:
            for bx, by, bz, u, v in obstacle_points:
                display[v, u] = (0, 0, 255)
            try:
                cv.imshow('Depth Obstacle', display)
                key = cv.waitKey(1)
                if key == 27:
                    rclpy.shutdown()
            except Exception:
                pass

        if obstacle_points:
            distances = [p[0] for p in obstacle_points]

            nearest = min(distances)

            self.get_logger().info(
                f'Obstacle points: '
                f'{len(obstacle_points)} | '
                f'Nearest: {nearest:.2f} m'
            )

            points = [
                (float(bx), float(by), float(bz)) 
                for bx, by, bz, _, _ in obstacle_points
            ]

            header = Header()
            header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
            header.frame_id = 'base_link'
            cloud = point_cloud2.create_cloud_xyz32(header, points)
            self.obstacle_pub.publish(cloud)

    def destroy_node(self):
        if self.show_gui:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass
        super().destroy_node()

def main():
    rclpy.init()
    node = DepthObstacleDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()