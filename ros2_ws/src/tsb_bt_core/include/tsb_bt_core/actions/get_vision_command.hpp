#ifndef TSB_BT_CORE__ACTIONS__GET_VISION_COMMAND_HPP_
#define TSB_BT_CORE__ACTIONS__GET_VISION_COMMAND_HPP_

#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "behaviortree_cpp/action_node.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace tsb_bt_core
{

/**
 * @brief BT Action: reads the latest classification published by the
 *        Python vision node (tsb_cv_brain) and exposes it on the blackboard.
 *
 * ROS interface
 *   Subscribes: /vision/command  (std_msgs/msg/String)  e.g. "LEFT", "RIGHT", "FORWARD"
 *
 * BT interface (suggested, feel free to change; keep tree.xml in sync)
 *   Output port "command" (std::string)
 *
 * Threading: the subscription callback runs on the executor thread started in
 * bt_executor.cpp, while tick() runs on the main thread. Always access
 * `latest_command_` while holding `mutex_`.
 */
class GetVisionCommand : public BT::SyncActionNode
{
public:
  static constexpr const char * kCommandTopic = "/vision/command";

  GetVisionCommand(
    const std::string & name, const BT::NodeConfig & config,
    rclcpp::Node::SharedPtr node)
  : BT::SyncActionNode(name, config), node_(std::move(node))
  {
    subscription_ = node_->create_subscription<std_msgs::msg::String>(
      kCommandTopic, rclcpp::QoS(10),
      [this](const std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_command_ = msg->data;
      });
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::OutputPort<std::string>(
        "command", "Latest command received from the vision node"),
      // TODO (candidate): add more ports if your design needs them.
    };
  }

  BT::NodeStatus tick() override
  {
    // =====================================================================
    // DONE (candidate): implement GetVisionCommand
    //   1. Read `latest_command_` (lock `mutex_` first).
    //   2. Decide what to return when no command has been received yet.
    //   3. Validate the value and write it to the "command" output port:
    //        setOutput("command", value);
    //   4. Return BT::NodeStatus::SUCCESS / FAILURE as appropriate.
    //
    // Tip: if this node should *wait* for a fresh command, derive from
    // BT::StatefulActionNode (onStart / onRunning / onHalted) instead.
    // =====================================================================
    //RCLCPP_WARN_THROTTLE(
    //  node_->get_logger(), *node_->get_clock(), 2000,
    //  "[%s] GetVisionCommand::tick() is not implemented yet", name().c_str());
    //return BT::NodeStatus::FAILURE;

    std::optional<std::string> command;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      command = latest_command_;
    }

    if (!command.has_value())
    {
      RCLCPP_INFO_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 3000,
        "[%s] waiting for comando on %s...", name().c_str(), kCommandTopic);
        return BT::NodeStatus::FAILURE;
    }

    const std::string & val = command.value();
    if (val != "FORWARD" && val != "BACKWARD" && val != "LEFT" &&
        val != "RIGHT" && val != "CIRCLE" )
    {
      RCLCPP_WARN(
        node_->get_logger(), "[%s] Ignoring invalid command: '%s'",
        name().c_str(), val.c_str());
      return BT::NodeStatus::FAILURE;
    }

    setOutput("command", val);
    return BT::NodeStatus::SUCCESS;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;

  std::mutex mutex_;
  std::optional<std::string> latest_command_;
};

}  // namespace tsb_bt_core

#endif  // TSB_BT_CORE__ACTIONS__GET_VISION_COMMAND_HPP_
