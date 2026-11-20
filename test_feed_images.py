from pathlib import Path
from PIL import Image as PILImage
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class StaticImageFeeder(Node):
    def __init__(self):
        super().__init__('static_image_feeder')
        self.pub = self.create_publisher(Image, '/vision/image', 10)

        root = Path('/home/joao/TSB/recrutamento2026/archive')
        self.samples = []
        for folder in ['Up', 'Left', 'Right', 'Down']:
            files = [p for p in root.rglob(f'{folder}/*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
            if files:
                self.samples.append((folder, files[0]))

        self.idx = 0
        self.timer = self.create_timer(2.5, self.timer_callback)

    def timer_callback(self):
        if not self.samples:
            self.get_logger().error('Nenhuma imagem encontrada em archive/')
            return

        folder, img_path = self.samples[self.idx % len(self.samples)]
        self.idx += 1

        arr = np.array(PILImage.open(img_path).convert('RGB'), dtype=np.uint8)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width, _ = arr.shape
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        msg.data = arr.tobytes()

        self.pub.publish(msg)
        self.get_logger().info(f'Sent static image [{folder}]: {img_path.name}')


def main():
    rclpy.init()
    node = StaticImageFeeder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
