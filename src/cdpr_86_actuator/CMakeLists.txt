cmake_minimum_required(VERSION 3.0.2)
project(cdpr_86_actuator)

find_package(catkin REQUIRED COMPONENTS
  cdpr_86_msg
  roscpp
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS cdpr_86_msg roscpp rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
