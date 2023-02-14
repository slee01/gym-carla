#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specified route.
"""


import carla
import math, itertools
import numpy as np

from shapely.geometry import Polygon

from agents.navigation.agent import Agent, AgentState
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import (get_speed, is_within_distance, get_trafficlight_trigger_location, compute_distance, positive)


class SafeAgent(Agent):
    """
    SafeAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    # def __init__(self, vehicle, target_speed=20):
    def __init__(self, vehicle, behavior='normal', dt=0.1, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(SafeAgent, self).__init__(vehicle)
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = target_speed
        self._min_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0
        self._dt = dt

        self._behavior = None
        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()        

        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']

        # self._proximity_threshold = 10.0  # meters
        # self._state = AgentState.NAVIGATING
        # args_lateral_dict = {
        #     'K_P': 1,
        #     'K_D': 0.02,
        #     'K_I': 0,
        #     'dt': 1.0/20.0}
        # self.local_planner = LocalPlanner(
        #     self._vehicle, opt_dict={'target_speed' : target_speed,
        #     'lateral_control_dict':args_lateral_dict})
        # self._hop_resolution = 2.0
        # self._path_seperation_hop = 2
        # self._path_seperation_threshold = 0.5
        # self._target_speed = target_speed
        # self._grp = None

        # Initialize the planners
        self.local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

        # Set waypoints and trajectories
        self.cand_trajs, self.pred_trajs = None, None
        self.cand_wpts, self.pred_wpts = None, None
        # self.waypoints = None
        self.desired_speeds = None
        
    # def add_emergency_stop(self, control):
    #     """
    #     Overwrites the throttle a brake values of a control to perform an emergency stop.
    #     The steering is kept the same to avoid going out of the lane when stopping during turns

    #         :param speed (carl.VehicleControl): control to be modified
    #     """
    #     control.throttle = 0.0
    #     control.brake = self._max_brake
    #     control.hand_brake = False
    #     return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
    
    def update_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self.local_planner.set_speed(speed)

    def update_follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self.local_planner.follow_speed_limits(value)

    def get_speed_limit(self):
        return self.local_planner._speed_limit

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self.local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_candidate_waypoints(self, types=None, length=50):
        """
        type: list of waypoint types such as PREDICTION, LANEFOLLOW, LANECHANGE, STOP, and GO
        length: length of each waypoint
        """
        assert types != None, "Waypoint type should be determined."

        self.cand_wpts = []
        for type in types:
            if type == "LANECHANGE":
                waypoints = self.lane_change(direction="left", same_lane_time=0.0, other_lane_time=0.0, lane_change_time=2.0)
            else:
                waypoints = self.local_planner.get_waypoints(length)
            self.cand_wpts.append({"type": type, "waypoints": waypoints})

    def set_candidate_trajectories(self, length=50, max_t=2.0):
        """Set trajectory (self.trajectory) to follow"""
        if self.cand_wpts is None:
            raise NotImplementedError

        if not self.cand_wpts:
            return []

        self.cand_trajs = []
        traj_length = int(max_t / self._dt)
        for candidate in self.cand_wpts:
            type, waypoints = candidate['type'], candidate['waypoints']
        
            speed = get_speed(self._vehicle) / 3.6  # TODO: replace current speed with target speed
            traj_gap = speed * self._dt
        
            trajectory = []
            trajectory.append(waypoints[0])

            left_dist, traj_count = 0.0, 0
            for i in range(len(waypoints)-1):
                left_dist += np.linalg.norm([waypoints[i][0]-waypoints[i+1][0], waypoints[i][1]-waypoints[i+1][1]])
                wp_vec_x, wp_vec_y = math.cos(math.radians(waypoints[i][2])), math.sin(math.radians(waypoints[i][2]))
                            
                while left_dist >= traj_gap:                
                    new_pt = [trajectory[-1][0] + wp_vec_x * traj_gap, trajectory[-1][1] + wp_vec_y * traj_gap, waypoints[i][2]]
                    left_dist -= traj_gap
                    trajectory.append(new_pt)
                    traj_count += 1
                    
                    if traj_count >= traj_length:
                        break
                
                if traj_count >= traj_length:
                    break
                
            self.cand_trajs.append(trajectory)

    def set_predicted_waypoints(self, length=50):
        self.pred_wpts = []
        waypoints = self.local_planner.get_waypoints(length)
        self.pred_wpts.append(waypoints)

    def set_predicted_trajectories(self, length=50, max_t=2.0):
        """Set trajectory (self.trajectory) to follow"""
        if self.pred_wpts is None:
            raise NotImplementedError

        if not self.pred_wpts:
            return []

        self.pred_trajs = []
        speed = get_speed(self._vehicle) / 3.6            
        traj_gap = speed * self._dt
        traj_length = int(max_t / self._dt)

        trajectory = []
        trajectory.append(self.pred_wpts[0])

        left_dist, traj_count = 0.0, 0
        for i in range(len(self.pred_wpts)-1):
            left_dist += np.linalg.norm([self.pred_wpts[i][0]-self.pred_wpts[i+1][0], self.pred_wpts[i][1]-self.pred_wpts[i+1][1]])
            wp_vec_x, wp_vec_y = math.cos(math.radians(self.pred_wpts[i][2])), math.sin(math.radians(self.pred_wpts[i][2]))
                        
            while left_dist >= traj_gap:                
                new_pt = [trajectory[-1][0] + wp_vec_x * traj_gap, trajectory[-1][1] + wp_vec_y * traj_gap, self.pred_wpts[i][2]]
                left_dist -= traj_gap
                trajectory.append(new_pt)
                traj_count += 1
                
                if traj_count >= traj_length:
                    break
            
            if traj_count >= traj_length:
                break
            
        self.pred_trajs.append(trajectory)

    # def get_trajectory(self, waypoints=None, speed=None, length=50, max_t=2.0):
    #     """Set trajectory (self.trajectory) to follow"""
    #     if waypoints is not None:
    #         # TODO: use waypoints to make different candidate waypoints for LaneChange scenario
    #         raise NotImplementedError
    #     if speed is None:
    #         speed = get_speed(self._vehicle) / 3.6
            
    #     traj_gap = speed * self._dt
    #     traj_length = int(max_t / self._dt)
        
    #     # self.set_waypoints(length)
        
    #     trajectory = []
    #     if not self.waypoints:
    #         # self.trajectory = []
    #         return trajectory
        
    #     # trajectory = []
    #     trajectory.append(self.waypoints[0])
    #     # print("=============================================================================")
    #     # print("traj_length: ", traj_length, " veh_speed: ", speed, " traj_gap: ", traj_gap)
    #     left_dist = 0.0
    #     traj_count = 0
    #     for i in range(len(self.waypoints)-1):
    #         left_dist += np.linalg.norm([self.waypoints[i][0]-self.waypoints[i+1][0], self.waypoints[i][1]-self.waypoints[i+1][1]])
    #         wp_vec_x, wp_vec_y = math.cos(math.radians(self.waypoints[i][2])), math.sin(math.radians(self.waypoints[i][2]))
                        
    #         while left_dist >= traj_gap:                
    #             new_pt = [
    #                 trajectory[-1][0] + wp_vec_x * traj_gap, trajectory[-1][1] + wp_vec_y * traj_gap, self.waypoints[i][2]]
    #             left_dist -= traj_gap
    #             trajectory.append(new_pt)
    #             traj_count += 1
    #             # print("-----------------------------------------------------------------------------")
    #             # print("i: ", i, " left_dist: ", left_dist, " traj_count: ", traj_count)
    #             # print("way_pt[i]: ", self.waypoints[i])
    #             # print("way_pt[i+1]: ", self.waypoints[i+1])
    #             # print("new_pt: ", new_pt)
                
    #             if traj_count >= traj_length:
    #                 break
            
    #         if traj_count >= traj_length:
    #             break
            
    #     # self.trajectory = trajectory
    #     # print("waypoint: ", self.waypoints)
    #     # print("trajectory: ", self.trajectory)
    #     return trajectory
    
    # def update_destination(self, location):
    def update_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        # if not start_location:
        #     start_location = self.local_planner.target_waypoint.transform.location
        #     clean_queue = True
        # else:
        #     start_location = self._vehicle.get_location()
        #     clean_queue = False

        start_location = self._vehicle.get_location()
        clean_queue = True

        # start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # end_waypoint = self._map.get_waypoint(carla.Location(location[0], location[1], location[2]))
        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        # print("-------------------------------------------------------------------")
        # print("vehicle id: ", self._vehicle.id, " location: ", self._vehicle.get_location())
        # print("start_waypoint: ", start_waypoint.transform.location)
        # print("end_waypoint: ", end_waypoint.transform.location)
        route_trace = self.trace_route(start_waypoint, end_waypoint)
        # print("route_trace: ", route_trace)
        self.local_planner.set_global_plan(route_trace, clean_queue=clean_queue)
        # assert route_trace

    def update_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self.local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """

        # # Setting up global router
        # if self._grp is None:
        #     dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
        #     grp = GlobalRoutePlanner(dao)
        #     grp.setup()
        #     self._grp = grp

        # # Obtain route plan
        # route = self._grp.trace_route(
        #     start_waypoint.transform.location,
        #     end_waypoint.transform.location)
        # return route

        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    # def run_step(self, debug=False):
    #     """
    #     Execute one step of navigation.
    #     :return: carla.VehicleControl
    #     """

    #     # is there an obstacle in front of us?
    #     hazard_detected = False

    #     # retrieve relevant elements for safe navigation, i.e.: traffic lights
    #     # and other vehicles
    #     actor_list = self._world.get_actors()
    #     vehicle_list = actor_list.filter("*vehicle*")
    #     lights_list = actor_list.filter("*traffic_light*")

    #     # check possible obstacles
    #     vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
    #     if vehicle_state:
    #         if debug:
    #             print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

    #         self._state = AgentState.BLOCKED_BY_VEHICLE
    #         hazard_detected = True

    #     # check for the state of the traffic lights
    #     light_state, traffic_light = self._is_light_red(lights_list)
    #     if light_state:
    #         if debug:
    #             print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

    #         self._state = AgentState.BLOCKED_RED_LIGHT
    #         hazard_detected = True

    #     if hazard_detected:
    #         control = self.emergency_stop()
    #     else:
    #         self._state = AgentState.NAVIGATING
    #         # standard local planner behavior
    #         control = self.local_planner.run_step(debug=debug)

    #     return control

    def run_step(self, debug=False):
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        # Car following behaviors
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        vehicle_state, vehicle, distance = self.get_hazard_obstacle(ego_vehicle_wp)
        
        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_control(vehicle, distance)

        # 3: Intersection behavior
        # elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
        #     target_speed = min([
        #         self._behavior.max_speed,
        #         self._speed_limit - 5])
        #     self._local_planner.set_speed(target_speed)
        #     control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            speed_limit = self.get_speed_limit()
            target_speed = min([self._behavior.max_speed, speed_limit - self._behavior.speed_lim_dist])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)        
        
        # control = self.local_planner.run_step()
        if hazard_detected:
            # control = self.add_emergency_stop(control)
            control = self.emergency_stop()

        return control

    def detect_hazard(self):
        """Detect hazardness of ego vehicle."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        return hazard_detected

    def done(self):
        """Check whether the agent has reached its destination."""
        return self.local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _generate_lane_change_path(
        self, 
        waypoint, 
        direction='left', 
        distance_same_lane=10,
        distance_other_lane=25, 
        lane_change_distance=25,
        check=True, 
        lane_changes=1, 
        step_distance=2):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self.local_planner.get_plan() is None:
            return (False, None, -1)

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            # r_vec = ego_transform.get_right_vector()
            # p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            # p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)

            forward_vec = ego_transform.get_forward_vector()
            r_vec_x = math.cos(math.radians(ego_transform.rotation.yaw - 90.0))
            r_vec_y = math.sin(math.radians(ego_transform.rotation.yaw - 90.0))
            # print("forward_vec: ", forward_vec, " r_vec_xy: ", r_vec_x, " ", r_vec_y)
            p1 = ego_location + carla.Location(r_ext * r_vec_x, r_ext * r_vec_y)
            p2 = ego_location + carla.Location(l_ext * r_vec_x, l_ext * r_vec_y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self.local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                # r_vec = wp.transform.get_right_vector()
                # p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                # p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                r_vec_x = math.cos(math.radians(ego_transform.rotation.yaw - 90.0))
                r_vec_y = math.sin(math.radians(ego_transform.rotation.yaw - 90.0))
                p1 = ego_location + carla.Location(r_ext * r_vec_x, r_ext * r_vec_y)
                p2 = ego_location + carla.Location(l_ext * r_vec_x, l_ext * r_vec_y)                
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            # if (use_bbs or target_wpt.is_junction) and route_polygon:
            if False:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())  # get_world_vertices is not exist in 0.9.6 CARLA
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self.local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)

    def get_target_speed(self, front_dist, front_speed, waypoint_type=None, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param front_vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if waypoint_type is None:
            raise NotImplementedError

        ego_speed = get_speed(self._vehicle)
        _ego_speed_limit = self.get_speed_limit()
        delta_v = max(1, (ego_speed - front_speed) / 3.6)
        ttc = front_dist / delta_v if delta_v != 0 else front_dist / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            # print("if self._behavior.safety_time > ttc > 0.0:")
            target_speed = min([positive(front_speed - self._behavior.speed_decrease), self._behavior.max_speed, _ego_speed_limit - self._behavior.speed_lim_dist])
        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            # print("elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:")
            target_speed = min([max(self._min_speed, front_speed), self._behavior.max_speed, _ego_speed_limit - self._behavior.speed_lim_dist])
        # Normal behavior.
        else:
            # print("else:")
            target_speed = min([self._behavior.max_speed, _ego_speed_limit - self._behavior.speed_lim_dist])

        GO_SPEED_LIMIT, STOP_SPEED_LIMIT = 10.0, 5.0
        if waypoint_type == "GO" and target_speed < GO_SPEED_LIMIT:
            target_speed = GO_SPEED_LIMIT
        elif waypoint_type == "STOP" and target_speed > STOP_SPEED_LIMIT:
            target_speed = STOP_SPEED_LIMIT

        if target_speed > ego_speed + self._behavior.speed_delta:
            target_speed = ego_speed + self._behavior.speed_delta
        elif target_speed < ego_speed - self._behavior.speed_delta:
            target_speed = ego_speed - self._behavior.speed_delta

        return target_speed

    def car_following_control(self, front_vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param front_vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        front_speed = get_speed(front_vehicle)
        ego_speed = get_speed(self._vehicle)
        _ego_speed_limit = self.get_speed_limit()
        delta_v = max(1, (ego_speed - front_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(front_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                _ego_speed_limit - self._behavior.speed_lim_dist])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, front_speed),
                self._behavior.max_speed,
                _ego_speed_limit - self._behavior.speed_lim_dist])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                _ego_speed_limit - self._behavior.speed_lim_dist])
            self.local_planner.set_speed(target_speed)
            control = self.local_planner.run_step(debug=debug)

        return control

    def get_hazard_obstacle(self, waypoint):
        """
        This module is in charge of warning in case of a collision.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        speed_limit = self.get_speed_limit()
        direction = self.local_planner.target_road_option
        if direction is None:
            direction = RoadOption.LANEFOLLOW

        if direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, speed_limit / 3), up_angle_th=30)

            # # Check for tailgating
            # if not vehicle_state and direction == RoadOption.LANEFOLLOW \
            #         and not waypoint.is_junction and self._speed > 10 \
            #         and self._behavior.tailgate_counter == 0:
            #     self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance
