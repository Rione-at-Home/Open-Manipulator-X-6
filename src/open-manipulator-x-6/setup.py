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
        ],
    },
)