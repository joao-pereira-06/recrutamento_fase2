#ifndef TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_
#define TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_

#include <chrono>
#include <string>
#include <utility>


#include "behaviortree_cpp/action_node.h"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include <algorithm>
#include <cmath>
#include "turtlesim/msg/pose.hpp"


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
  static constexpr const char * kPoseTopic = "/turtle1/pose";

  MoveTurtle(
    const std::string & name, const BT::NodeConfig & config,
    rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(std::move(node))
  {
    cmd_vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>(
      kCmdVelTopic, rclcpp::QoS(10));

    pose_sub_ = node_->create_subscription<turtlesim::msg::Pose>(
      kPoseTopic, rclcpp::QoS(10),
      [this](const turtlesim::msg::Pose::SharedPtr msg) {
        current_theta_ = msg->theta;
        has_pose_ = true;
      });
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>(
        "command", "FORWARD", "Direction to move: FORWARD | BACKWARD | LEFT | RIGHT | CIRCLE"),
      BT::InputPort<double>("speed", 1.5, "Linear or angular speed"),
      BT::InputPort<double>("duration", 1.0, "Duration of the movement in seconds")
    };
  }

  BT::NodeStatus onStart() override
  {
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
    start_time_ = node_->now();
    start_theta_ = current_theta_;

    RCLCPP_INFO(node_->get_logger(), "[%s] Starting %s", name().c_str(), current_cmd_.c_str());
    return onRunning();
  }

  BT::NodeStatus onRunning() override
  {
    // --- Rotação exata de 90 graus (pi/2 rad) usando /turtle1/pose ---
    if ((current_cmd_ == "LEFT" || current_cmd_ == "RIGHT") && has_pose_)
    {
      constexpr double kTargetAngle = M_PI / 2.0;  // 90 graus exatos
      double turned = std::abs(normalizeAngle(current_theta_ - start_theta_));
      double remaining = kTargetAngle - turned;

      if (remaining <= 0.005)  // Tolerância < 0.3 graus
      {
        stopTurtle();
        RCLCPP_INFO(node_->get_logger(), "[%s] Finished %s (exact 90 deg)", name().c_str(), current_cmd_.c_str());
        return BT::NodeStatus::SUCCESS;
      }

      // Controlo proporcional: roda rápido no início e abranda perto dos 90° para não passar
      double angular_vel = std::clamp(remaining * 4.0, 0.15, speed_);
      geometry_msgs::msg::Twist twist;
      twist.angular.z = (current_cmd_ == "LEFT") ? angular_vel : -angular_vel;
      cmd_vel_pub_->publish(twist);
      return BT::NodeStatus::RUNNING;
    }

    // --- Movimentos lineares / círculo por duração ---
    double elapsed = (node_->now() - start_time_).seconds();
    if (elapsed >= duration_sec_)
    {
      stopTurtle();
      RCLCPP_INFO(node_->get_logger(), "[%s] Finished %s", name().c_str(), current_cmd_.c_str());
      return BT::NodeStatus::SUCCESS;
    }

    publishLinearOrCircle();
    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override
  {
    stopTurtle();
  }

private:
  static double normalizeAngle(double angle)
  {
    return std::atan2(std::sin(angle), std::cos(angle));
  }

  void publishLinearOrCircle()
  {
    geometry_msgs::msg::Twist twist;
    if (current_cmd_ == "FORWARD") {
      twist.linear.x = speed_;
    } else if (current_cmd_ == "BACKWARD") {
      twist.linear.x = -speed_;
    } else if (current_cmd_ == "CIRCLE") {
      twist.linear.x = speed_;
      twist.angular.z = speed_;
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
  rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr pose_sub_;

  std::string current_cmd_;
  double speed_{1.5};
  double duration_sec_{1.0};
  rclcpp::Time start_time_;

  double current_theta_{0.0};
  double start_theta_{0.0};
  bool has_pose_{false};
};
}  // namespace tsb_bt_core

#endif  // TSB_BT_CORE__ACTIONS__MOVE_TURTLE_HPP_