#!/usr/bin/env python
"""
ROS node for controlling a TurtleBot3 in Gazebo for reinforcement learning.

This script defines a Gazebo simulation environment interface for training
a robot navigation agent. It handles robot state retrieval (odometry, laser scans),
applying actions (velocity commands), resetting the simulation to new states,
and calculating rewards based on proximity to a target and obstacles.
"""

import rospy
import rospkg
import tf
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import os
import threading
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel

from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
import time
from geometry_msgs.msg import Pose
import numpy as np
import math
import random

from std_srvs.srv import Empty


class InfoGetter(object):
    """
    Helper class to receive messages asynchronously and block until a message is received.
    """

    def __init__(self):
        # Event that will block until the info is received
        self._event = threading.Event()
        # Attribute for storing the received message
        self._msg = None

    def __call__(self, msg):
        """
        Uses __call__ so the object itself acts as the callback.
        Save the data, trigger the event.
        """
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """
        Blocks until the data is received with optional timeout.
        Returns the received message.
        """
        self._event.wait(timeout)
        return self._msg


def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles to a quaternion."""

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


class GameState:
    """
    Represents the state and interface for the Gazebo simulation environment
    for a single TurtleBot agent.
    """

    def __init__(self, id):
        """
        Initializes the GameState for a specific robot ID.

        Args:
            id (int): The unique identifier for the robot.
        """
        self.id = id
        self.cmd_ig = InfoGetter()
        self.pose_ig = InfoGetter()
        self.laser_ig = InfoGetter()
        self.collision_ig = InfoGetter()

        # Global constants for topic names and frame IDs
        self.PUBLISHER_TOPIC_NAME = "/cmd_vel"
        self.SUBSCRIBER_TOPIC_CMD_NAME = "/cmd_vel"
        self.SUBSCRIBER_TOPIC_LASER_NAME = "/laserscan_filtered"
        self.ODOM_FRAME_NAME = "/odom"
        self.BASE_FRAME_FOOTPRINT = "/base_footprint"
        self.BASE_FRAME_LINK = "/base_link"
        self.GAZEBO_MODEL_STATES_TOPIC = "/gazebo/model_states"
        self.GAZEBO_SET_MODEL_STATE_SERVICE = "/gazebo/set_model_state"
        self.GAZEBO_RESET_SIMULATION_SERVICE = "gazebo/reset_simulation"

        self.pub = rospy.Publisher(self.PUBLISHER_TOPIC_NAME, Twist, queue_size=1)
        self.position = Point()
        self.move_cmd = Twist()
        self.action = np.zeros((1, 2))

        # Subscribers
        self.cmd_info = rospy.Subscriber(self.SUBSCRIBER_TOPIC_CMD_NAME, Twist, self.cmd_ig)
        self.pose_info = rospy.Subscriber(self.GAZEBO_MODEL_STATES_TOPIC, ModelStates, self.pose_ig)
        self.laser_info = rospy.Subscriber(self.SUBSCRIBER_TOPIC_LASER_NAME, LaserScan, self.laser_ig)

        # TF listener for robot pose
        self.tf_listener = tf.TransformListener()
        self.odom_frame = self.ODOM_FRAME_NAME
        # Determine the base frame dynamically
        try:
            self.tf_listener.waitForTransform(self.odom_frame, self.BASE_FRAME_FOOTPRINT, rospy.Time(), rospy.Duration(1.0))
            self.base_frame = self.BASE_FRAME_FOOTPRINT
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, self.BASE_FRAME_LINK, rospy.Time(), rospy.Duration(1.0))
                self.base_frame = self.BASE_FRAME_LINK
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")

        rospy.loginfo(f"Using base frame: {self.base_frame}")
        # Get initial pose
        (self.position, self.rotation) = self.get_odom()

        self.goal_position = Pose()
        self.rate = rospy.Rate(100)  # 100hz

        # Create a Twist message and add linear x and angular z values (default values)
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.6  # linear_x
        self.move_cmd.angular.z = 0.2  # angular_z



        # observation_space and action_space dimensions
        # 24 laser values + distance to target + angle to target + linear velocity + angular velocity = 28
        self.state_num = 28
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)  # Represents the structure/size of the observation
        self.action_space = np.empty(self.action_num)  # Represents the structure/size of the action

        self.laser_reward = 0
        # set target position (initial values, will be reset)
        self.target_x = 10
        self.target_y = 10

        # set turtlebot index in gazebo world (needs to be correct for ModelStates message)
        self.model_index = 10  # 25

        # Counter for detecting if the robot is stuck or rotating excessively
        self.count = 0

        # Service proxies for Gazebo
        self.reset_proxy = rospy.ServiceProxy(self.GAZEBO_RESET_SIMULATION_SERVICE, Empty)
        self.set_state_proxy = rospy.ServiceProxy(self.GAZEBO_SET_MODEL_STATE_SERVICE, SetModelState)


    def get_odom(self):
        """
        Gets the robot's current position and orientation from the TF tree.

        Returns:
            tuple: A tuple containing a Point object for the position and
                   the yaw angle (rotation around Z) in radians.
            None: If a TF exception occurs.
        """
        try:
            # Lookup the transform from the odom frame to the robot's base frame
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            # Convert the quaternion rotation to Euler angles
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception in get_odom")
            return None

        # Return the position (x, y, z) and the yaw angle (rotation[2])
        return (Point(*trans), rotation[2])

    def shutdown(self):
        """Stops the robot by publishing a zero velocity command."""
        self.pub.publish(Twist())
        rospy.sleep(1)

    def print_odom(self):
        """Prints the robot's current position and rotation (yaw)."""
        (self.position, self.rotation) = self.get_odom()
        rospy.loginfo("position is %s, %s, %s", self.position.x, self.position.y, self.position.z,)
        rospy.loginfo("rotation is %s", self.rotation)

    def reset(self):
        """
        Resets the robot and target positions in the Gazebo simulation
        and returns the initial state.
        """

        # Reset internal state variables
        self.count = 0

        # Define base offsets for robot initial position based on ID
        offset_x = 0
        offset_y = 0
        if self.id == 0:
            offset_x = -10
            offset_y = -10
        elif self.id == 1:
            offset_x = 10
            offset_y = -10
        elif self.id == 2:
            offset_x = -10
            offset_y = 10
        elif self.id == 3:
            offset_x = 10
            offset_y = 10
        elif self.id == 4:
            offset_x = -10
            offset_y = 30
        elif self.id == 5:
            offset_x = 10
            offset_y = 30

        # Generate random target position relative to the base offset
        index_list = [-1, 1]
        index_x = random.choice(index_list)
        index_y = random.choice(index_list)
        # Random deviation from a point 7 units away in a random quadrant relative to offset
        self.target_x = (np.random.random() - 0.5) * 1.5 + index_x * 7 + offset_x
        self.target_y = (np.random.random() - 0.5) * 1.5 + index_y * 7 + offset_y

        # Prepare ModelState messages for setting robot and target positions
        state_msg = ModelState()
        state_msg.model_name = "tb3_" + str(self.id)
        state_msg.pose.position.x = offset_x
        state_msg.pose.position.y = offset_y
        state_msg.pose.position.z = 0.0
        # Set initial orientation (facing roughly along X axis with a slight deviation)
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = -0.2
        state_msg.pose.orientation.w = 0

        state_target_msg = ModelState()
        # Assuming the target model name follows this pattern
        state_target_msg.model_name = "unit_sphere_" + str(self.id) + "_0_0_0"
        state_target_msg.pose.position.x = self.target_x
        state_target_msg.pose.position.y = self.target_y
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = -0.2
        state_target_msg.pose.orientation.w = 0

        # Wait for the Gazebo services to be available
        rospy.wait_for_service(self.GAZEBO_RESET_SIMULATION_SERVICE)
        rospy.wait_for_service(self.GAZEBO_SET_MODEL_STATE_SERVICE)

        # Set the state of the robot and target models
        try:
            # self.reset_proxy() # This is commented out, meaning the simulation is not fully reset
            resp = self.set_state_proxy(state_msg)
            resp_target = self.set_state_proxy(state_target_msg)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e) # Changed print to rospy.logerr

        # Update the stored goal position
        self.goal_position.position.x = self.target_x
        self.goal_position.position.y = self.target_y

        # Set initial velocities to zero and publish
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        # Give Gazebo time to update
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()

        # Construct the initial state observation (mostly placeholders initially)
        initial_state = np.ones(self.state_num)
        # Set the last few elements (linear_x, angular_z, distance, angle_diff) to 0
        initial_state[self.state_num - 1] = 0
        initial_state[self.state_num - 2] = 0
        initial_state[self.state_num - 3] = 0
        initial_state[self.state_num - 4] = 0

        return initial_state

    def get_laser_values(self):
        """
        Retrieves the latest laser scan values.

        Returns:
            list: A list of range values from the laser scanner.
        """
        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        # normalized_laser = [(x)/3.5 for x in (laser_msg.ranges)] # This normalized value is calculated again in read_states/game_step
        return laser_values

    def turtlebot_is_crashed(self, laser_values, range_limit):
        """
        Checks if the robot has crashed based on laser readings within a limit.

        Args:
            laser_values (list): The list of laser range values.
            range_limit (float): The minimum distance to consider a crash.

        Returns:
            float: A negative reward value if crashed, 0 otherwise.
        """
        self.laser_crashed_value = 0  # Seems unused?
        self.laser_crashed_reward = 0

        for i in range(len(laser_values)):
            # Check for close proximity (potential collision warning)
            if laser_values[i] < 2 * range_limit:
                self.laser_crashed_reward = -80
            # Check for actual collision (very close proximity)
            if laser_values[i] < range_limit:
                self.laser_crashed_value = 1  # Indicate a crash occurred
                self.laser_crashed_reward = -200  # Assign a large negative reward
                self.reset()  # Reset the environment upon crash
                rospy.loginfo("crashed!!!!!!!!!!!!!!!!") # Changed print to rospy.loginfo
                time.sleep(1)  # Pause briefly after reset
                break  # Stop checking once a crash is detected
        return self.laser_crashed_reward

    def collision_detection(self, laser_values, range_limit):
        """
        Detects if any laser reading is within the specified range limit.

        Args:
            laser_values (list): The list of laser range values.
            range_limit (float): The maximum distance to consider as a collision.

        Returns:
            int: 1 if collision is detected, 0 otherwise.
        """
        self.collision_detected = 0
        for i in range(len(laser_values)):
            if laser_values[i] < range_limit:
                self.collision_detected = 1
                break
        return self.collision_detected



    def read_cmd(self):
        """
        Reads the latest commanded linear and angular velocities.

        Returns:
            tuple: A tuple containing the linear_x and angular_z velocities.
        """
        cmd_msg = self.cmd_ig.get_msg()
        linear_x = cmd_msg.linear.x
        angular_z = cmd_msg.angular.z
        return linear_x, angular_z
  
    def read_states(self):
        """
        Reads the current state of the robot including laser data, pose,
        distance and angle to target, and current velocities.

        Returns:
            np.ndarray: A numpy array representing the current state observation.
        """
        # Get current command velocities
        linear_x, angular_z = self.read_cmd()

        # Get current robot pose
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        angle_turtlebot = self.rotation

        # Calculate the angle between the robot's current heading and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

        # Normalize angles to be within [0, 2*pi)
        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2 * math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2 * math.pi

        # Calculate the difference between the robot's angle and the target angle,
        # normalized to be within (-pi, pi]
        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2 * math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2 * math.pi

        # Prepare the normalized laser value
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        # Normalize laser readings by a maximum expected range (3.5m)
        normalized_laser = [x / 3.5 for x in laser_msg.ranges]

        # Calculate the current distance to the target
        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x) ** 2 + (self.target_y - turtlebot_y) ** 2)

        # Construct the state vector
        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        # Append the current commanded velocities to the state
        state = np.append(state, linear_x)
        state = np.append(state, angular_z)

        # Reshape the state to be a row vector (1, state_num)
        state = state.reshape(1, self.state_num)
        return state

    def game_step(self, time_step=0.1, linear_x=0.8, angular_z=0.3):
        """
        Executes a single step in the simulation by applying the given velocities,
        calculating the new state, and determining the reward.

        Args:
            time_step (float): The duration of the step (although time.sleep(0.1) is used).
            linear_x (float): The linear velocity command to apply (before scaling).
            angular_z (float): The angular velocity command to apply.

        Returns:
            tuple: A tuple containing:
                   - reward (float): The reward for the current step.
                   - state (np.ndarray): The new state observation.
                   - laser_crashed_value (int): 1 if crashed, 0 otherwise.
        """

        # Scale linear velocity (using a factor of 0.26)
        self.move_cmd.linear.x = linear_x * 0.26
        self.move_cmd.angular.z = angular_z

        rospy.loginfo(f"linear: {self.move_cmd.linear.x}")  # Changed print to rospy.loginfo
        rospy.loginfo(f"angle: {self.move_cmd.angular.z}")  # Changed print to rospy.loginfo

        # Get robot's pose before applying the action
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x_previous = self.position.x
        turtlebot_y_previous = self.position.y

        # Publish the velocity command
        self.pub.publish(self.move_cmd)
        # Wait for a fixed duration to allow the robot to move
        time.sleep(0.1)

        # Get robot's pose after applying the action
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        angle_turtlebot = self.rotation

        # Calculate the angle between the turtlebot's current heading and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

        # Normalize angles to be within [0, 2*pi)
        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2 * math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2 * math.pi

        # Calculate the difference between the robot's angle and the target angle,
        # normalized to be within (-pi, pi]
        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2 * math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2 * math.pi

        rospy.loginfo(f"angle_diff: {angle_diff}")  # Changed print to rospy.loginfo

        # Prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        rospy.loginfo(f"laser values: {laser_values}")  # Changed print to rospy.loginfo
        # Normalize laser readings by a maximum expected range (3.5m)
        normalized_laser = [x / 3.5 for x in laser_msg.ranges]

        # Calculate the current distance to the target
        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x) ** 2 + (self.target_y - turtlebot_y) ** 2)

        # Construct the state vector
        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        # Append the current commanded velocities to the state (scaled linear_x)
        state = np.append(state, linear_x * 0.26)
        state = np.append(state, angular_z)
        # Reshape the state to be a row vector (1, state_num)
        state = state.reshape(1, self.state_num)

        # Calculate the distance reward (difference in distance to target)
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous) ** 2 + (self.target_y - turtlebot_y_previous) ** 2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x) ** 2 + (self.target_y - turtlebot_y) ** 2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target

        # Check for collision and get collision reward
        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        # Calculate a reward based on the sum of normalized laser readings (encourages staying away from obstacles)
        # Subtracting 24 suggests a baseline or penalty for having small readings
        self.laser_reward = sum(normalized_laser) - 24
        # Total collision-related reward
        self.collision_reward = self.laser_crashed_reward + self.laser_reward

        # Penalize high angular velocities
        self.angular_punish_reward = 0
        if angular_z > 0.5:
            self.angular_punish_reward = -100
        if angular_z < -0.5:
            self.angular_punish_reward = -100

        # Penalize very low linear velocities
        self.linear_punish_reward = 0
        if linear_x * 0.26 < 0.15:  # Check the scaled linear velocity
            self.linear_punish_reward = -50

        # Check if the robot has reached the target
        self.arrive_reward = 0
        if current_distance_turtlebot_target < 0.8:
            self.arrive_reward = 400  # Large positive reward for reaching target
            self.reset()  # Reset the environment upon reaching target
            rospy.loginfo("arrive!!!!!!!!!!!!!!!!") # Changed print to rospy.loginfo
            time.sleep(1)  # Pause briefly after reset

        ###########################################################################
        # Check for getting stuck or rotating excessively
        # This condition (distance_reward*(5/time_step)*1.2*7 < 5) seems complex and possibly
        # related to forward progress per time step. If progress is too slow, increment count.
        # If count exceeds a threshold (85), reset.
        if distance_reward * (5 / time_step) * 1.2 * 7 < 5:
            self.count = self.count + 1
            if self.count > 85:  # 85
                rospy.loginfo("rotate too much!!!!!!!!!!") # Changed print to rospy.loginfo
                self.reset()
                time.sleep(1)

        ###########################################################################

        # Calculate the total reward for the step
        # The distance reward is scaled significantly
        reward = (distance_reward * (5 / time_step) * 1.2 * 7 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward)
        rospy.loginfo("laser_reward is %s", self.laser_reward)  # Changed print to rospy.loginfo
        rospy.loginfo("laser_crashed_reward is %s", self.laser_crashed_reward)  # Changed print to rospy.loginfo
        rospy.loginfo("arrive_reward is %s", self.arrive_reward)  # Changed print to rospy.loginfo
        rospy.loginfo("distance reward is : %s", distance_reward * (5 / time_step) * 1.2 * 7)  # Changed print to rospy.loginfo

        # Return the calculated reward, the new state, and the crash indicator
        return reward, state, self.laser_crashed_value