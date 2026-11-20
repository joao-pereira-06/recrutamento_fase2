// bt_executor: loads a BehaviorTree.CPP v4 tree from XML and ticks it,
// giving the tree's action nodes access to a shared rclcpp::Node.

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/groot2_publisher.h"
#include "behaviortree_cpp/xml_parsing.h"
#include "rclcpp/rclcpp.hpp"

#include "tsb_bt_core/actions/get_vision_command.hpp"
#include "tsb_bt_core/actions/move_turtle.hpp"

using namespace std::chrono_literals;

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // ----------------------------------------------------------------- ROS node
  auto node = std::make_shared<rclcpp::Node>("bt_executor");

  const std::string default_tree_file =
    ament_index_cpp::get_package_share_directory("tsb_bt_core") +
    "/behavior_trees/tree.xml";

  const std::string tree_file =
    node->declare_parameter<std::string>("tree_file", default_tree_file);
  const double tick_rate_hz = node->declare_parameter<double>("tick_rate_hz", 10.0);
  const int groot_port = node->declare_parameter<int>("groot_port", 1667);
  const bool print_models = node->declare_parameter<bool>("print_tree_nodes_model", false);

  // Spin the node in a background thread so subscription callbacks keep
  // running while the main thread ticks the tree.
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {executor.spin();});

  // ------------------------------------------------------- Behavior Tree setup
  BT::BehaviorTreeFactory factory;

  // Extra constructor arguments (here: the rclcpp node) are forwarded to the
  // action node constructors.
  factory.registerNodeType<tsb_bt_core::GetVisionCommand>("GetVisionCommand", node);
  factory.registerNodeType<tsb_bt_core::MoveTurtle>("MoveTurtle", node);

  // TODO (candidate): register any additional custom nodes here
  // (Actions, Conditions, Decorators...).

  if (print_models) {
    // Paste this output into the <TreeNodesModel> of tree.xml so Groot2 knows
    // about the ports you declared in C++.
    RCLCPP_INFO(
      node->get_logger(), "TreeNodesModel:\n%s",
      BT::writeTreeNodesModelXML(factory).c_str());
  }

  RCLCPP_INFO(node->get_logger(), "Loading behavior tree: %s", tree_file.c_str());

  BT::Tree tree;
  try {
    tree = factory.createTreeFromFile(tree_file);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(node->get_logger(), "Failed to create tree: %s", e.what());
    executor.cancel();
    spin_thread.join();
    rclcpp::shutdown();
    return 1;
  }

  // Live monitoring: open Groot2 -> "Monitor" tab -> connect to localhost:<groot_port>
  BT::Groot2Publisher groot_publisher(tree, static_cast<unsigned>(groot_port));

  // ------------------------------------------------------------ execution loop
  rclcpp::Rate rate(tick_rate_hz);
  while (rclcpp::ok()) {
    // Ticks the tree until it returns SUCCESS or FAILURE.
    // NOTE: if your nodes can stay RUNNING forever, make them check
    // rclcpp::ok() so Ctrl+C still stops the program.
    const BT::NodeStatus status = tree.tickWhileRunning(10ms);
    /*
    if (status == BT::NodeStatus::SUCCESS)
    {
      RCLCPP_INFO(node->get_logger(), "Drawing completed successfully! Stopping executor.");
      break;
    }
    */
    rclcpp::Rate rate(tick_rate_hz);
    while (rclcpp::ok())
    {
      const BT::NodeStatus status = tree.tickWhileRunning(10ms);

      RCLCPP_DEBUG(
      node->get_logger(), "Tree finished with status: %s",
      BT::toStr(status).c_str());

    rate.sleep();
    }
    
    // TODO (candidate): decide what happens when the tree finishes.
    // (Keep looping? Stop on SUCCESS? Log FAILURE reasons?)
    RCLCPP_DEBUG(
      node->get_logger(), "Tree finished with status: %s",
      BT::toStr(status).c_str());

    rate.sleep();
  }

  // ------------------------------------------------------------------ cleanup
  tree.haltTree();
  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
