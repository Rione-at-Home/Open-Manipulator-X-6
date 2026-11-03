import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'open_manipulator_x6'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ri-one',
    maintainer_email='dev@todo.todo',
    description='ROS 2 driver for 6-DOF OpenManipulator-X arm.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'arm_driver = open_manipulator_x6.arm_driver:main',
            'session_logger = open_manipulator_x6.session_logger:main',
            'keyboard_teleop = open_manipulator_x6.keyboard_teleop:main',
            "teach_pendant = open_manipulator_x6.teach_pendant:main",
            "sequence_runner = open_manipulator_x6.sequence_runner:main",
<<<<<<< HEAD
            "aruco_detector = open_manipulator_x6.aruco_detector:main",
            "camera_calibrator = open_manipulator_x6.camera_calibrator:main",
=======
>>>>>>> f9e072399d2f7e32ad73d0713eb448fc9a5dbf36
        ],
    },
)