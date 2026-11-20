cmake_minimum_required(VERSION 3.16)
project(tsb_bt_core LANGUAGES CXX)

# BehaviorTree.CPP v4 requires C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------ dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)

set(dependencies
  rclcpp
  geometry_msgs
  std_msgs
  behaviortree_cpp
  ament_index_cpp
)

# ------------------------------------------------------------------ executables
add_executable(bt_executor src/bt_executor.cpp)
target_include_directories(bt_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(bt_executor ${dependencies})

# TODO (candidate): if you add more .cpp files (e.g. one per BT node),
# list them in add_executable() above or build them into a library.

# ------------------------------------------------------------------ install
install(TARGETS bt_executor
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Behavior Tree XML files -> share/tsb_bt_core/behavior_trees
# Launch files            -> share/tsb_bt_core/launch
# With `colcon build --symlink-install` these are symlinks to src/, so edits
# made in Groot2 are picked up without rebuilding.
install(DIRECTORY behavior_trees launch
  DESTINATION share/${PROJECT_NAME}
)

# ------------------------------------------------------------------ tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip checks that are noisy for a challenge skeleton
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_dependencies(${dependencies})
ament_package()
