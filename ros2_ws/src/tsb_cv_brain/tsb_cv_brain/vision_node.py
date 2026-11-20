"""
Vision node: classifies incoming images into turtle commands with PyTorch.

ROS interface
    Subscribes: /vision/image    (sensor_msgs/msg/Image)
    Publishes:  /vision/command  (std_msgs/msg/String), one of COMMANDS

The C++ Behavior Tree (tsb_bt_core, GetVisionCommand node) listens on
/vision/command, so keep the topic name and the command strings in sync.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
import torch.nn as nn
import torch.nn.functional as F

# Contract with the C++ Behavior Tree: these are the only strings it expects.
COMMANDS = ('BACKWARD', 'LEFT', 'RIGHT', 'FORWARD')


class TurtleVisionCNN(nn.Module):
    """CNN that maps an image to one of the COMMANDS classes."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        # entrada (B, 3, 224, 224)
        self.features = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class VisionNode(Node):
    """ROS 2 wrapper around TurtleVisionCNN."""

    def __init__(self):
        super().__init__('vision_node')

        # ------------------------------------------------------ parameters
        self.declare_parameter('image_topic', '/vision/image')
        self.declare_parameter('command_topic', '/vision/command')
        self.declare_parameter('model_path', '')

        image_topic = self.get_parameter('image_topic').value
        command_topic = self.get_parameter('command_topic').value
        model_path = self.get_parameter('model_path').value

        # ----------------------------------------------------------- model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'PyTorch {torch.__version__}, device: {self.device}')

        self.model = TurtleVisionCNN().to(self.device)
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.get_logger().info(f'Loaded trained weights from "{model_path}"')
            except Exception as e:
                self.get_logger().error(f'Failed to load weights from "{model_path}": {e}')
        else:
            self.get_logger().warn('No model_path given, the model is untrained')
            self.model.eval()

        # ------------------------------------------- publisher / subscriber
        self.command_pub = self.create_publisher(String, command_topic, 10)
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'Listening on "{image_topic}", publishing commands on "{command_topic}"')

    def image_callback(self, msg: Image) -> None:
        """Classify one image and publish the resulting command."""
        try:
            tensor = self.preprocess(msg).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            command = self.postprocess(logits)
            self.get_logger().info(f'Classified arrow as: {command}')
            self.publish_command(command)
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')

    def preprocess(self, msg: Image) -> torch.Tensor:
        """Convert a sensor_msgs/Image into a model input tensor (1, 3, 224, 224)."""
        if msg.encoding in ('rgb8', 'bgr8'):
            channels = 3
        elif msg.encoding == 'mono8':
            channels = 1
        else:
            raise ValueError(f'Unsupported image encoding: {msg.encoding}')

        # Respeita o stride (msg.step) caso exista padding nas linhas
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        img = raw[:, :msg.width * channels].reshape((msg.height, msg.width, channels))

        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1].copy()
        elif channels == 1:
            img = np.repeat(img, 3, axis=2)

        # Converte para (1, 3, H, W) em [0.0, 1.0] (igual ao transforms.ToTensor() do treino)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        )
        return tensor

    def postprocess(self, logits: torch.Tensor) -> str:
        """Convert model logits into one of COMMANDS."""
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return COMMANDS[pred_idx]

    def publish_command(self, command: str) -> None:
        """Publish a command for the Behavior Tree."""
        if command not in COMMANDS:
            self.get_logger().error(f'Refusing to publish unknown command "{command}"')
            return
        self.command_pub.publish(String(data=command))


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()