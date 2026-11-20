"""
Vision node: classifies incoming images (ROS topic or Web Canvas HTTP) with PyTorch.

ROS interface
    Subscribes: /vision/image    (sensor_msgs/msg/Image)
    Publishes:  /vision/command  (std_msgs/msg/String), one of COMMANDS
    Web UI:     http://localhost:8080
"""

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
import torch.nn as nn
import torch.nn.functional as F

COMMANDS = ('BACKWARD', 'LEFT', 'RIGHT', 'FORWARD')
WEB_INDEX = Path('/home/joao/TSB/recrutamento2026/web/index.html')


class TurtleVisionCNN(nn.Module):
    """CNN that maps an image to one of the COMMANDS classes."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

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
    """ROS 2 wrapper around TurtleVisionCNN + Web Canvas HTTP server."""

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('image_topic', '/vision/image')
        self.declare_parameter('command_topic', '/vision/command')
        self.declare_parameter('model_path', '')
        self.declare_parameter('web_port', 8080)

        image_topic = self.get_parameter('image_topic').value
        command_topic = self.get_parameter('command_topic').value
        model_path = self.get_parameter('model_path').value
        web_port = int(self.get_parameter('web_port').value)

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

        self.command_pub = self.create_publisher(String, command_topic, 10)
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        self._start_web_server(web_port)
        self.get_logger().info(
            f'Listening on "{image_topic}" & Web UI http://localhost:{web_port}')

    def _start_web_server(self, port: int) -> None:
        vision_node = self

        class CanvasHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/', '/index.html') and WEB_INDEX.exists():
                    content = WEB_INDEX.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_error(404, 'Not Found')

            def do_POST(self):
                if self.path != '/predict':
                    self.send_error(404, 'Not Found')
                    return

                length = int(self.headers.get('Content-Length', 0))
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                b64_data = payload['image'].split(',', 1)[1]
                img_bytes = base64.b64decode(b64_data)

                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
                command = vision_node.classify_pil_image(pil_img)
                vision_node.get_logger().info(f'[Web Canvas] Classified arrow as: {command}')
                vision_node.publish_command(command)

                resp = json.dumps({'command': command}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, format, *args):
                pass  # Evita poluir o log do ROS 2 com pedidos HTTP

        server = HTTPServer(('0.0.0.0', port), CanvasHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    def classify_pil_image(self, pil_img: PILImage.Image) -> str:
        pil_resized = pil_img.resize((224, 224))
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        return self.postprocess(logits)

    def image_callback(self, msg: Image) -> None:
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
        if msg.encoding in ('rgb8', 'bgr8'):
            channels = 3
        elif msg.encoding == 'mono8':
            channels = 1
        else:
            raise ValueError(f'Unsupported image encoding: {msg.encoding}')

        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        img = raw[:, :msg.width * channels].reshape((msg.height, msg.width, channels))

        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1].copy()
        elif channels == 1:
            img = np.repeat(img, 3, axis=2)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        )
        return tensor

    def postprocess(self, logits: torch.Tensor) -> str:
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return COMMANDS[pred_idx]

    def publish_command(self, command: str) -> None:
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