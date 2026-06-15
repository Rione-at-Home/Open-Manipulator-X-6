cmake_minimum_required(VERSION 3.8)
project(complete_arm_assembly)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  urdf
  meshes
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()