import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_bot_navigation = get_package_share_directory("bot_navigation")

    nav2_params = os.path.join(
        pkg_bot_navigation,
        "config",
        "nav2_params.yaml"
    )

    Planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[nav2_params]
    )

    controller = Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[
                nav2_params,
                {
                    "enable_stamped_cmd_vel": True
                }
            ],
            remappings=[
                ("cmd_vel", "/diff_drive_controller/cmd_vel")
            ]
        )

    bt = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[nav2_params]
    )

    behavior = Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[nav2_params]
        )

    global_costmap = Node(
            package="nav2_costmap_2d",
            executable="costmap_server",
            name="global_costmap",
            output="screen",
            parameters=[nav2_params]
        )

    local_costmap = Node(
            package="nav2_costmap_2d",
            executable="costmap_server",
            name="local_costmap",
            output="screen",
            parameters=[nav2_params]
        )

    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            nav2_params , 
            {
                "autostart": True,
                "node_names": [
                    "planner_server",
                    "controller_server",
                    "bt_navigator",
                    "behavior_server",
                ],
            },
        ]
    )

    return LaunchDescription([
        Planner,
        controller, 
        bt,
        behavior,
        # global_costmap,
        # local_costmap,
        lifecycle
    ])