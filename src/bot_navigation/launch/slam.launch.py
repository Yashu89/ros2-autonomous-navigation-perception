import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch_ros.events.lifecycle import ChangeState, matches_node_name
from lifecycle_msgs.msg import Transition
from launch.event_handlers import OnProcessStart
from launch_ros.event_handlers import OnStateTransition 

def generate_launch_description():
    pkg_bot_description = get_package_share_directory("bot_navigation")

    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    declare_use_sim_time = DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
        )
    
    slam_params = os.path.join(
        pkg_bot_description,
        "config",
        "map_params_async.yaml"
    )

    slam_toolbox = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        namespace="",
        output="screen",
        parameters=[
            slam_params,
            {"use_sim_time": use_sim_time}
        ]
    )

    configure_slam = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_node_name("/slam_toolbox"),
            transition_id=Transition.TRANSITION_CONFIGURE
        )
    )

    activate_slam = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_node_name("/slam_toolbox"),
            transition_id=Transition.TRANSITION_ACTIVATE
        )
    )

    configure_slam_startup = RegisterEventHandler(
        OnProcessStart(
            target_action=slam_toolbox,
            on_start=[
                configure_slam
            ]
        )
    ) 

    activate_slam_startup = RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=slam_toolbox,
                goal_state="inactive",
                entities=[
                    activate_slam
                ]
            )
        )

    return LaunchDescription([
        declare_use_sim_time,
        slam_toolbox,
        configure_slam_startup,
        activate_slam_startup,
    ])