#ifndef TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_
#define TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_

#include <chrono>
#include <string>
#include <utility>

#include "behaviortree_cpp/action_node.h"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

namespace tsb_bt_core
{

/**
 * @brief BT Action: turns a high-level command into a velocity command for
 *        the turtle.
 *
 * ROS interface
 *   Publishes: /turtle1/cmd_vel  (geometry_msgs/msg/Twist)
 *
 * BT interface (suggested, feel free to change; keep tree.xml in sync)
 *   Input port "command" (std::string)  e.g. "LEFT", "RIGHT", "FORWARD"
 */
class MoveTurtle : public BT::StatefulActionNode
{
public:
  static constexpr const char * kCmdVelTopic = "/turtle1/cmd_vel";

  MoveTurtle(
    const std::string & name, const BT::NodeConfig & config,
    rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(std::move(node))
  {
    cmd_vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>(
      kCmdVelTopic, rclcpp::QoS(10));
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>(
        "command", "FORWARD", "Direction to move: FORWARD | BACKWARD | LEFT | RIGHT | CIRCLE"),
      // DONE (candidate): add ports for tuning, e.g. linear/angular speed.
      BT::InputPort<double>("speed", 1.5, "Linear or anguular speed"),
      BT::InputPort<double>("duration", 1.0, "Duration of the movement is seconds")
    };
  }

  BT::NodeStatus onStart() override
  {
    // =====================================================================
    // DONE (candidate): implement MoveTurtle
    //   1. Read the "command" input port:
    //        BT::Expected<std::string> cmd = getInput<std::string>("command");
    //      and handle the error case (`if (!cmd) { ... cmd.error() ... }`). v
    //   2. Map the command to a geometry_msgs::msg::Twist
    //      (linear.x / angular.z).v
    //   3. Publish it with `cmd_vel_pub_->publish(twist);`v
    //   4. Return BT::NodeStatus::SUCCESS / FAILURE as appropriate.v
    //
    // Tip: to move for a fixed time/distance, derive from
    // BT::StatefulActionNode and return RUNNING until the motion is done
    // (remember to stop the turtle in onHalted()).
    // =====================================================================

    auto cmd_res = getInput<std::string>("command");
    auto speed_res = getInput<double>("speed");
    auto dur_res = getInput<double>("duration");

    if (!cmd_res || !speed_res || !dur_res)
    {
      RCLCPP_ERROR(node_->get_logger(), "[%s] Missing required Blackboard inputs!", name().c_str());
      return BT::NodeStatus::FAILURE;
    }
    
    current_cmd_ = cmd_res.value();
    speed_ = speed_res.value();
    duration_sec_ = dur_res.value();
    start_time_ = node_-> now();

    RCLCPP_INFO(node_->get_logger(), "[%s] Starting %s (speed: %.2f, dur: %.2fs)",
                name().c_str(), current_cmd_.c_str(), speed_, duration_sec_);

  //   RCLCPP_WARN_THROTTLE(
  //     node_->get_logger(), *node_->get_clock(), 2000,
  //     "[%s] MoveTurtle::tick() is not implemented yet", name().c_str());
  //   return BT::NodeStatus::FAILURE;
  // }

    publishVelocity();
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    double elapsed = (node_->now() - start_time_).seconds();

    if (elapsed >= duration_sec_)
    {
      stopTurtle();
      RCLCPP_INFO(node_->get_logger(), "[%s] Finished %s", name().c_str(), current_cmd_.c_str());
      return BT::NodeStatus::SUCCESS;
    }

    publishVelocity();
    return BT::NodeStatus::RUNNING;
  }


  void onHalted() override
  {stopTurtle();}


private:
  void publishVelocity()
  {
    geometry_msgs::msg::Twist twist;

    if (current_cmd_ == "FORWARD") {
      twist.linear.x = speed_;
    } else if (current_cmd_ == "BACKWARD") {
      twist.linear.x = -speed_;
    } else if (current_cmd_ == "LEFT") {
      twist.angular.z = speed_;
    } else if (current_cmd_ == "RIGHT") {
      twist.angular.z = -speed_;
    } else if (current_cmd_ == "CIRCLE") {
      twist.linear.x = speed_;
      twist.angular.z = speed_;
    } else {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "[%s] Unknown command '%s', stopping turtle.", name().c_str(), current_cmd_.c_str());
    }

    cmd_vel_pub_->publish(twist);
  }

  void stopTurtle()
  {
    geometry_msgs::msg::Twist stop_msg;
    cmd_vel_pub_->publish(stop_msg);
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

  std::string current_cmd_;
  double speed_{1.0};
  double duration_sec_{1.0};
  rclcpp::Time start_time_;

}; // namespace tsb_bt_core

}

#endif  // TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_
