"""Launch turtlesim, the PyTorch vision node and the Behavior Tree executor."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_tree = os.path.join(
        get_package_share_directory('tsb_bt_core'), 'behavior_trees', 'tree.xml')

    tree_file = LaunchConfiguration('tree_file')
    model_path = LaunchConfiguration('model_path')
    groot_port = LaunchConfiguration('groot_port')

    return LaunchDescription([
        DeclareLaunchArgument(
            'tree_file', default_value=default_tree,
            description='Behavior Tree XML executed by bt_executor'),
        DeclareLaunchArgument(
            'model_path', default_value='',
            description='Path to the trained PyTorch weights (.pt/.pth)'),
        DeclareLaunchArgument(
            'groot_port', default_value='1667',
            description='Port used by Groot2 for live monitoring'),

        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim',
            output='screen',
        ),
        Node(
            package='tsb_cv_brain',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{'model_path': ParameterValue(model_path, value_type=str)}],
        ),
        Node(
            package='tsb_bt_core',
            executable='bt_executor',
            name='bt_executor',
            output='screen',
            parameters=[{
                'tree_file': ParameterValue(tree_file, value_type=str),
                'groot_port': ParameterValue(groot_port, value_type=int),
            }],
        ),
    ])
