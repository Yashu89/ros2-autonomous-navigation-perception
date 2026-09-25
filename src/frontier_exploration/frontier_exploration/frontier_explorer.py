import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.last_cluster_sizes = None
        # Check if pursuing any goal
        self.goal_active = False

        # Blacklisting & failure tracking
        self.failed_goals = []  # List of (x, y) tuples for failed goals
        self.cluster_failure_counts = {}  # Map region -> count
        self.current_goal = None

        self.get_logger().info('Frontier Exploration Node Started')

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        data = msg.data

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
        except tf2_ros.TransformException:
            return

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        # Identify frontier cells (free cells bordering unknown space -1)
        frontier_cells = set()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                index = y * width + x
                if data[index] != 0:
                    continue

                neighbours = [
                    (x - 1, y - 1),
                    (x, y - 1),
                    (x + 1, y - 1),
                    (x - 1, y),
                    (x + 1, y),
                    (x - 1, y + 1),
                    (x, y + 1),
                    (x + 1, y + 1),
                ]

                is_frontier = False
                for nx, ny in neighbours:
                    neighbour_index = ny * width + nx
                    if data[neighbour_index] == -1:
                        is_frontier = True
                        break
                if is_frontier:
                    frontier_cells.add((x, y))

        # Clustering connected frontier cells
        clusters = []
        visited = set()

        for start_cell in frontier_cells:
            if start_cell in visited:
                continue

            cluster = []
            queue = [start_cell]
            visited.add(start_cell)

            while queue:
                current_x, current_y = queue.pop(0)
                cluster.append((current_x, current_y))

                neighbours = [
                    (current_x - 1, current_y - 1),
                    (current_x, current_y - 1),
                    (current_x + 1, current_y - 1),
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x - 1, current_y + 1),
                    (current_x, current_y + 1),
                    (current_x + 1, current_y + 1),
                ]

                for nx, ny in neighbours:
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) in frontier_cells and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

            clusters.append(cluster)

        # Centroids and Candidate Goal Selection
        min_frontier_size = 4
        frontier_threshold = 0.8

        cluster_info = []

        for cluster in clusters:
            cells_count = len(cluster)
            if cells_count < min_frontier_size:
                continue

            total_x = sum(x for x, y in cluster)
            total_y = sum(y for x, y in cluster)

            centroid_x = total_x / cells_count
            centroid_y = total_y / cells_count

            threshold_cells = frontier_threshold / resolution

            candidate_frontier_cells = [
                (x, y)
                for x, y in cluster
                if ((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
                < threshold_cells**2
            ]

            candidate_frontier_count = len(candidate_frontier_cells)

            # Search known free space near frontier (0.2m to 0.6m pullback)
            min_pullback = max(1, int(0.2 / resolution))
            max_pullback = int(0.6 / resolution)

            candidate_free_cells = set()
            for fx, fy in candidate_frontier_cells:
                for dy in range(-max_pullback, max_pullback + 1):
                    for dx in range(-max_pullback, max_pullback + 1):
                        dist_sq = dx * dx + dy * dy
                        if (min_pullback * min_pullback <= dist_sq <= max_pullback * max_pullback):
                            nx, ny = fx + dx, fy + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                n_idx = ny * width + nx
                                if data[n_idx] == 0:  # Known free space
                                    candidate_free_cells.add((nx, ny))

            candidate_free_cells = list(candidate_free_cells)
            candidate_free_count = len(candidate_free_cells)

            # Filter candidates for safety:
            safe_candidate_cells = []
            for candidate_x, candidate_y in candidate_free_cells:
                if self.is_goal_safe(
                    candidate_x,
                    candidate_y,
                    data,
                    width,
                    height,
                    resolution,
                    obstacle_radius=0.25,
                    unknown_radius=0.10,
                ):
                    safe_candidate_cells.append((candidate_x, candidate_y))

            safe_candidate_count = len(safe_candidate_cells)

            if not safe_candidate_cells:
                for candidate_x, candidate_y in candidate_free_cells:
                    if self.is_goal_safe(
                        candidate_x,
                        candidate_y,
                        data,
                        width,
                        height,
                        resolution,
                        obstacle_radius=0.18,
                        unknown_radius=0.08,
                    ):
                        safe_candidate_cells.append((candidate_x, candidate_y))

            # Select best candidate goal closest to cluster centroid
            candidate_goal = None
            sorted_safe_cells = sorted(
                safe_candidate_cells,
                key=lambda cell: (
                    (cell[0] - centroid_x) ** 2 + (cell[1] - centroid_y) ** 2
                ),
            )

            for cell in sorted_safe_cells:
                cell_world_x = origin_x + (cell[0] + 0.5) * resolution
                cell_world_y = origin_y + (cell[1] + 0.5) * resolution
                if not self.is_goal_blacklisted(cell_world_x, cell_world_y):
                    candidate_goal = cell
                    break

            centroid_world_x = origin_x + (centroid_x + 0.5) * resolution
            centroid_world_y = origin_y + (centroid_y + 0.5) * resolution

            candidate_world_x = None
            candidate_world_y = None

            if candidate_goal is not None:
                candidate_x, candidate_y = candidate_goal
                candidate_world_x = origin_x + (candidate_x + 0.5) * resolution
                candidate_world_y = origin_y + (candidate_y + 0.5) * resolution

            cluster_info.append({
                'size': cells_count,
                'centroid': (centroid_world_x, centroid_world_y),
                'candidate_goal': (candidate_world_x, candidate_world_y),
                'candidate_frontier_count': candidate_frontier_count,
                'candidate_free_count': candidate_free_count,
                'safe_candidate_count': safe_candidate_count,
            })

        # Scoring and ranking of clusters
        for cluster in cluster_info:
            candidate_x, candidate_y = cluster['candidate_goal']

            if candidate_x is None or candidate_y is None:
                cluster['distance'] = None
                cluster['score'] = float('-inf')
                continue

            distance = math.hypot(candidate_x - robot_x, candidate_y - robot_y)
            cluster['distance'] = distance

            if distance < 0.35:
                cluster['score'] = float('-inf')
                continue

            # Look up failure count for this region
            failures = self.get_region_failures(cluster['centroid'][0], cluster['centroid'][1])

            size_factor = math.pow(cluster['size'], 0.6)
            dist_factor = max(distance, 0.4)
            penalty = 1.0 + 3.0 * failures
            cluster['score'] = size_factor / (dist_factor * penalty)

        valid_clusters = [
            cluster
            for cluster in cluster_info
            if cluster['candidate_goal'][0] is not None
            and cluster['score'] > float('-inf')
            and not self.is_goal_blacklisted(
                cluster['candidate_goal'][0], cluster['candidate_goal'][1]
            )
        ]

        valid_clusters.sort(key=lambda c: c['score'], reverse=True)

        best_cluster = None
        best_goal = None
        best_centroid = None

        if valid_clusters:
            best_cluster = valid_clusters[0]
            best_goal = best_cluster['candidate_goal']
            best_centroid = best_cluster['centroid']

        # Dispatch Goal
        if best_goal is not None and not self.goal_active:
            # Point yaw toward the goal position (the safe pullback cell), NOT the centroid.
            # Centroid can be inside/behind a wall → atan2(centroid) forces the bot to
            # spin toward the wall at goal arrival, causing the aggressive overshoot+correct loop.
            yaw = math.atan2(
                best_goal[1] - robot_y, best_goal[0] - robot_x
            )
            self.send_goal(best_goal[0], best_goal[1], yaw, best_centroid)

        # Logging output
        cluster_sizes = [cluster['size'] for cluster in valid_clusters]
        if cluster_sizes != self.last_cluster_sizes:
            self.get_logger().info(
                f'Frontier Cells: {len(frontier_cells)} | Clusters:'
                f' {len(clusters)} | Valid Candidates: {len(valid_clusters)}'
            )
            for i, cluster in enumerate(cluster_info):
                candidate_x, candidate_y = cluster['candidate_goal']
                c_x, c_y = cluster['centroid']
                score_str = (f"{cluster['score']:.2f}"
                    if cluster.get('score') is not None 
                    else 'nothing')
                self.get_logger().info(f"Cluster {i+1}: {cluster['size']} cells | Centroid:({c_x:.2f}, {c_y:.2f}) | Score: {score_str}")
                if candidate_x is not None:
                    self.get_logger().info(f'  Goal: ({candidate_x:.2f}, {candidate_y:.2f}) m')
                else:
                    self.get_logger().info('  Goal: NONE (Unreachable/Unsafe)')
            self.last_cluster_sizes = cluster_sizes

    def is_goal_safe(
        self,
        goal_x,
        goal_y,
        data,
        width,
        height,
        resolution,
        obstacle_radius=0.25,
        unknown_radius=0.10,
    ):
        """Checks if a goal cell has sufficient clearance from solid obstacles and known space."""
        obstacle_cells = max(1, int(obstacle_radius / resolution))
        unknown_cells = max(1, int(unknown_radius / resolution))

        max_cells = max(obstacle_cells, unknown_cells)

        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                dist_sq = dx * dx + dy * dy

                check_x = goal_x + dx
                check_y = goal_y + dy

                if (
                    check_x < 0
                    or check_x >= width
                    or check_y < 0
                    or check_y >= height
                ):
                    return False

                index = check_y * width + check_x
                val = data[index]

                # Check walls / obstacles
                if dist_sq <= obstacle_cells * obstacle_cells:
                    if val >= 50:  # Occupied / obstacle
                        return False

                # Known free space check around target pose
                if dist_sq <= unknown_cells * unknown_cells:
                    if val == -1: 
                        return False

        return True

    def is_goal_blacklisted(self, goal_x, goal_y):
        """Checks if a candidate goal is within 0.5m of any failed goal."""
        blacklist_tolerance = 0.5  # 50 cm blacklist radius for failed goals

        for failed_x, failed_y in self.failed_goals:
            distance = math.hypot(goal_x - failed_x, goal_y - failed_y)
            if distance < blacklist_tolerance:
                return True
        return False

    def record_failed_goal(self, goal):
        """Records a failed goal and updates region failure count."""
        if goal is None:
            return
        self.failed_goals.append(goal)
        self.get_logger().warn(f'Blacklisted Failed Goal: ({goal[0]:.2f}, {goal[1]:.2f}) [Radius: 0.5m]')

        key = (round(goal[0], 0), round(goal[1], 0))
        self.cluster_failure_counts[key] = (
            self.cluster_failure_counts.get(key, 0) + 1
        )

    def get_region_failures(self, cx, cy):
        """Gets number of failures recorded near region centroid."""
        key = (round(cx, 0), round(cy, 0))
        return self.cluster_failure_counts.get(key, 0)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2 server.')
            if self.current_goal is not None:
                self.record_failed_goal(self.current_goal)
            self.goal_active = False
            self.current_goal = None
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        result = future.result()
        if result.status == 4:
            self.get_logger().info('Navigation Succeeded!')
            if self.current_goal is not None:
                key = (round(self.current_goal[0], 0), round(self.current_goal[1], 0))
                self.cluster_failure_counts.pop(key, None)
        elif result.status == 5:
            self.get_logger().warn('Navigation Canceled.')
        elif result.status == 6:
            self.get_logger().warn('Navigation Failed (Aborted).')
            if self.current_goal is not None:
                self.record_failed_goal(self.current_goal)
        else:
            self.get_logger().warn(f'Navigation finished with status: {result.status}')  

        self.goal_active = False
        self.current_goal = None

    def send_goal(self, goal_x, goal_y, yaw=0.0, centroid=None):
        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.position.z = 0.0

        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav.wait_for_server()
        self.goal_active = True
        self.current_goal = (goal_x, goal_y)
        self.get_logger().info(
            f'Sending Goal: ({goal_x:.2f}, {goal_y:.2f}) | Heading:'
            f' {math.degrees(yaw):.1f}°'
        )

        send_goal_future = self.nav.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)


def main():
    rclpy.init()
    node = FrontierExplorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()