import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    pkg_bot_description = get_package_share_directory("bot_description")
    xacro_file = os.path.join(pkg_bot_description, "urdf", "bot.urdf.xacro")
    default_world_path = os.path.join(pkg_bot_description, "worlds", "fetchit_challenge_tests.world")

    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    world_file = LaunchConfiguration("world", default=default_world_path)

    robot_description = xacro.process_file(xacro_file).toxml()

    gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.pathsep.join([
            pkg_bot_description,
            os.path.join(pkg_bot_description, "world_models"),
            os.path.dirname(pkg_bot_description)
        ])
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true"
    )

    declare_world_cmd = DeclareLaunchArgument(
        "world",
        default_value=default_world_path,
        description="Full path to world file"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            )
        ),
        launch_arguments={
            "gz_args": [ "-r ", world_file ]
        }.items()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time
            }
        ],
        output="screen"
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "bot",
            "-z", "0.09228"
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    diff_drive_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    controllers_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                joint_state_broadcaster,
                diff_drive_controller
            ]
        )
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU"
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/image_raw/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/image_raw/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/camera/image_raw/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        ],
        output='screen',
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world_cmd,
        gz_resource_path,
        gazebo,
        ros_gz_bridge,
        camera_bridge,
        robot_state_publisher,
        spawn_robot,
        controllers_after_spawn
    ])