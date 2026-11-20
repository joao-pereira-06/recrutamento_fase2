from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'tsb_cv_brain'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Trained weights placed in models/ are installed to
        # share/tsb_cv_brain/models (rebuild after adding new files).
        (os.path.join('share', package_name, 'models'),
            glob('models/*.pt') + glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TSB Recruitment',
    maintainer_email='alexandre.oneill@tecnico.ulisboa.pt',
    description='PyTorch vision node that turns images into turtle commands.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = tsb_cv_brain.vision_node:main',
        ],
    },
)
