import time
import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from ultralytics import YOLO

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

from vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        self.model = YOLO('/home/yash/perception/yolo11n.pt')

        self.image_sub = self.create_subscription(Image, '/camera/image_raw/image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/image_raw/depth_image', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/image_raw/camera_info', self.camera_info_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.detect_pub = self.create_publisher(Detection2DArray, '/perception/detections', 10)
        self.detect_3d_pub = self.create_publisher(Detection3DArray, '/perception/detections_3d', 10)

        self.last_time = time.perf_counter()
        self.fps = 0.0

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.get_logger().info('Perception Node Started')

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]

        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def pose_3d(self, center_x, center_y):
        if self.latest_depth is None:
            return None
        if self.fx is None:
            return None

        height, width = self.latest_depth.shape

        if not (0 <= center_x < width and 0 <= center_y < height):
            return None

        radius = 3

        x1 = max(0, center_x - radius)
        x2 = min(width, center_x + radius + 1)

        y1 = max(0, center_y - radius)
        y2 = min(height, center_y + radius + 1)

        depth_region = self.latest_depth[y1:y2, x1:x2]

        valid_depth = depth_region[depth_region > 0]

        if len(valid_depth) == 0:
            return None

        #depth estimation
        z = float(np.median(valid_depth))

        x = (center_x - self.cx) * z / self.fx
        y = (center_y - self.cy) * z / self.fy

        return x, y, z
        
    def transform_point(self, x, y, z, source_frame, target_frame):
        point = PointStamped()

        point.header.frame_id = source_frame
        point.header.stamp = self.get_clock().now().to_msg()

        point.point.x = x
        point.point.y = y
        point.point.z = z

        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            transformed = do_transform_point(point , transform)
            return(
                transformed.point.x,
                transformed.point.y,
                transformed.point.z
            )
        except Exception as e:
            self.get_logger().warn(f'TF {source_frame} to {target_frame} failed: {e}')
            return None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        detection_msg = Detection2DArray()
        detection_msg.header = msg.header

        detection_3d_msg = Detection3DArray()
        detection_3d_msg.header = msg.header

        results = self.model(frame, conf=0.25, verbose=False)

        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                center_x = int((x1 + x2)/2)
                center_y = int((y1 + y2)/2)

                detection = Detection2D()

                detection.bbox.center.position.x = float(center_x)
                detection.bbox.center.position.y = float(center_y)

                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                hypothesis = ObjectHypothesisWithPose()

                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)
                detection_msg.detections.append(detection)

                pose = self.pose_3d(center_x, center_y)
                if pose is not None:
                    x, y, z = pose

                    base_pose = self.transform_point(x, y, z, 'camera_link', 'base_link')

                    if base_pose is not None:
                        bx, by, bz = base_pose

                        # map_pose = self.transform_point(bx, by, bz, 'base_link', 'map')

                        # if map_pose is not None:
                        #     mx, my, mz = map_pose

                        #     detection_3d = Detection3D()
                        #     detection_3d.bbox.center.position.x = float(mx)
                        #     detection_3d.bbox.center.position.y = float(my)
                        #     detection_3d.bbox.center.position.z = float(mz)
                        #     detection_3d.results.append(hypothesis)
                        #     detection_3d_msg.detections.append(detection_3d)

                        self.get_logger().info(f'{class_name} conf={confidence:.2f}' 
                                           f'center=({center_x}, {center_y})'
                                           f'camera=({x:.2f}, {y:.2f}, {z:.2f})'
                                           f'base=({bx:.2f}, {by:.2f}, {bz:.2f})'
                                        #    f'map=({mx:.2f}, {my:.2f}, {mz:.2f})'
                                           )


                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                label = f'{class_name} {confidence:.2f}'

                cv.putText(frame, label, (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.detect_pub.publish(detection_msg)
        self.detect_3d_pub.publish(detection_3d_msg)

        current_time = time.perf_counter()
        dt = current_time - self.last_time

        if dt > 0:
            self.fps = 1.0/dt
        self.last_time = current_time

        cv.putText(frame, f'FPS:{self.fps:.1f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.imshow('YOLO Perception', frame)
        cv.waitKey(1)

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()