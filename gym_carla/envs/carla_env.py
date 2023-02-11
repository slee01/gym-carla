#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
# import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

# from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from agents.navigation.safe_agent import SafeAgent


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.dt = params['dt']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.d_behind = params['d_behind']
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']

    # self.pred_dt = params['pred_dt']
    # self.pred_time = params['pred_time']
    self.pred_dt = self.dt * 10.0
    self.pred_time = 5.0
    
    self.dests = None
    self.collision_infos = None
    self.discrete, self.discrete_act = None, None
    self.n_acc, self.n_steer = None, None
    self.action_space = None
    self.observation_space = None

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(10.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.vehicles = []

    # Create the ego vehicle blueprint
    # self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='51,255,255')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    self.collision_sensor = None

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
  def reset(self):
    # Clear sensor objects
    if self.collision_sensor is not None:
      self.collision_sensor.stop()
      self.collision_sensor.destroy()
    self.collision_sensor = None
    self._clear_all_vehicles()
    self.vehicles = []
    
    # Delete sensors, vehicles and walkers
    # self._clear_all_actors(['sensor.other.collision', 'vehicle.*'])
    # self._clear_all_actors(['vehicle.*'])
    # Disable sync mode
    self._set_synchronous_mode(False)
    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        # if np.linalg.norm([spawn_point.location.x - 52.1, spawn_point.location.y - 4.2]) < 10.0:
          # continue
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      spawn_point = random.choice(self.vehicle_spawn_points)
      # if np.linalg.norm([spawn_point.location.x - 52.1, spawn_point.location.y - 4.2]) < 10.0:
      #     continue
      if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
        count -= 1

    self._set_random_vehicle_paths()

    # Get actors polygon list
    self.vehicle_polygons = []
    # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    vehicle_poly_dict = self._get_vehicle_polygons()
    self.vehicle_polygons.append(vehicle_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'random':
        transform = random.choice(self.vehicle_spawn_points)
      if self.task_mode == 'roundabout':
        self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        transform = set_carla_transform(self.start)
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)
        
    self._set_ego_vehicle_path()
    self._set_vehicle_waypoints_and_trajectory()
    
    # print("ego_location: ", self.ego.get_location())
    # for i, vehicle in enumerate(self.vehicles):
    #   print("i: ", i, " vehicle location: ", vehicle.get_location())

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego._vehicle)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
        
    self.collision_hist = []

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    # print("Enable Sync Mode & Apply Settings")
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # print("Set All Vehicle Waypoints and Trajectories")
    # self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    ############################################################################
    # DEFINE WAYPOINTS AND VEHICLE_FRONT(HAZARD) HERE
    # self.waypoints = self.ego.local_planner.get_waypoints(length=50)
    self._set_vehicle_waypoints_and_trajectory()
    
    # print("Detect Ego Vehicle Hazard")
    self.vehicle_front =  self.ego.detect_hazard()
    # print("self.waypoints: ", self.waypoints, " length: ", len(self.waypoints))
    # print("self.vehicle_front: ", self.vehicle_front)
    ############################################################################
    
    # print("Complete Reset")
    return self._get_obs()
  
  def step(self, action):
    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    # print("Apply Network Control to Ego Vehicle")
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)
    
    # print("Apply Analytic Control to Ego Vehicle")
    self._apply_random_vehicle_control()

    self.world.tick()

    # Append actors polygon list
    # print("Append Actors Polygon List")
    # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    vehicle_poly_dict = self._get_vehicle_polygons()
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)

    # route planner
    # print("Set All Vehicle Waypoints and Trajectories")
    # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    self._set_vehicle_waypoints_and_trajectory()
    # self.waypoints = self.ego.local_planner.get_waypoints(length=50)
    # print("Detect Ego Vehicle Hazard")
    self.vehicle_front =  self.ego.detect_hazard()

    # print("Get Time and Dist to Collision")
    _collision_infos = self._get_random_vehicle_time_and_dist_to_collision()
    self.collision_infos = sorted(_collision_infos, key=lambda d: d['time_to_collision'], reverse=False)
    # print("collision_infos: ", self.collision_infos)
    
    # state information
    info = {
      'waypoints': self.ego.waypoints,
      'vehicle_front': self.vehicle_front
    }
    
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def _get_obs(self):
    raise NotImplementedError
    # """Get the observations."""

    # # State observation
    # ego_trans = self.ego.get_transform()
    # ego_x = ego_trans.location.x
    # ego_y = ego_trans.location.y
    # ego_yaw = ego_trans.rotation.yaw/180*np.pi
    # lateral_dis, w = get_preview_lane_dis(self.ego.waypoints, ego_x, ego_y)
    # delta_yaw = np.arcsin(np.cross(w, 
    #   np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    # v = self.ego.get_velocity()
    # speed = np.sqrt(v.x**2 + v.y**2)
      
    # state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
    # return state
  
  def _get_reward(self):
    raise NotImplementedError
    # """Calculate the step reward."""
    # # reward for speed tracking
    # v = self.ego.get_velocity()
    # speed = np.sqrt(v.x**2 + v.y**2)
    # r_speed = -abs(speed - self.desired_speed)
    
    # # reward for collision
    # r_collision = 0
    # if len(self.collision_hist) > 0:
    #   r_collision = -1

    # # reward for steering:
    # r_steer = -self.ego.get_control().steer**2

    # # reward for out of lane
    # ego_x, ego_y = get_pos(self.ego)
    # dis, w = get_lane_dis(self.ego.waypoints, ego_x, ego_y)
    # r_out = 0
    # if abs(dis) > self.out_lane_thres:
    #   r_out = -1

    # # longitudinal speed
    # lspeed = np.array([v.x, v.y])
    # lspeed_lon = np.dot(lspeed, w)

    # # cost for too fast
    # r_fast = 0
    # if lspeed_lon > self.desired_speed:
    #   r_fast = -1

    # # cost for lateral acceleration
    # r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    # r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

    # return r
  
  def _terminal(self):
    raise NotImplementedError
    # """Calculate whether to terminate the current episode."""
    # # Get ego state
    # ego_x, ego_y = get_pos(self.ego)

    # # If collides
    # if len(self.collision_hist)>0: 
    #   return True
    # # If reach maximum timestep
    # if self.time_step>=self.max_time_episode:
    #   return True
    # # If at destination
    # if self.dests is not None: # If at destination
    #   for dest in self.dests:
    #     if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
    #       return True
    # # If out of lane
    # dis, _ = get_lane_dis(self.ego.waypoints, ego_x, ego_y)
    # if abs(dis) > self.out_lane_thres:
    #   return True
    # return False
    
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
      # bp.set_attribute('color', '255,255,255')
    return bp

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      # vehicle.set_autopilot()
      vehicle = SafeAgent(vehicle, behavior="normal", dt=self.pred_dt, target_speed=20.0)
      self.vehicles.append(vehicle)
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=SafeAgent(vehicle, behavior="normal", dt=self.pred_dt, target_speed=20.0)
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def _get_vehicle_polygons(self):
    """Get the bounding box polygon of vehicles.

    Args:
    Returns:
      vehicle_poly_dict: a dictionary containing the bounding boxes of vehicles.
    """
    vehicle_poly_dict={}
    for vehicle in self.vehicles:
      # Get x, y and yaw of the vehicle
      trans=vehicle.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=vehicle.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the vehicle's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      vehicle_poly_dict[vehicle.id]=poly
    return vehicle_poly_dict

  def _set_random_vehicle_paths(self):
    if self.dests is not None: # If at destination
      for vehicle in self.vehicles:
        # print("self.dests: ", self.dests)  # [[x,y,z], [x,y,z], [x,y,z], ..., [x,y,z]]
        _dest = random.choice(self.dests)
        # vehicle.set_destination(random.choice(self.dests))
        vehicle.set_destination(end_location=carla.Location(_dest[0], _dest[1], _dest[2]))
        # print("vehicle.id: ", vehicle._vehicle.id, " location: ", vehicle.get_location(), "destinaiton: ", _dest)
        # print("vehicle.local_planner.get_waypoints(): ", vehicle.local_planner.get_waypoints())
        # print("vehicle.local_planner._waypoint_buffer: ", vehicle.local_planner._waypoint_buffer)
    else:
      raise NotImplementedError

  def _set_ego_vehicle_path(self):
    if self.ego is not None and self.dests is not None: 
      _dest = random.choice(self.dests)
      self.ego.set_destination(end_location=carla.Location(_dest[0], _dest[1], _dest[2]))
    else:
      raise NotImplementedError

  def _set_vehicle_waypoints_and_trajectory(self, max_t=5.0):
    # print("===================================================")
    # print("===================================================")
    # print("Ego Vechie: ", self.ego.id, " speed: ", get_speed(self.ego) / 3.6, " Ego Location: ", self.ego.get_location())
    self.ego.set_trajectory(max_t=max_t)
    # print("Ego Traj: ", self.ego.trajectory)
    for vehicle in self.vehicles:
      vehicle.set_trajectory(max_t=max_t)
      # print("---------------------------------------------------")
      # print("Vehicle: ", vehicle.id, " speed: ", get_speed(vehicle) / 3.6, " Location: ", vehicle.get_location())
      # print("Traj: ", vehicle.trajectory)
      
  def _get_random_vehicle_time_and_dist_to_collision(self):
    collisions_list = []
    for trajectory in self.ego.trajectories:
      collisions = []
      for vehicle in self.vehicles:
        if vehicle.is_alive:
          collision = self._get_time_to_collision(trajectory, vehicle, max_time=self.pred_time)
          collisions.append(collision)
          
      collisions_list.append(collisions)
                
    return collisions_list
  
  def _get_time_to_collision(self, trajectory, vehicle, buf_t = 2.0, max_time=5.0, max_dist=80.0):
    if len(trajectory) < 2 or len(vehicle.trajectory) < 2:
      return {"id": vehicle.id, "time_to_collision": max_time / max_time, "dist_to_collision": max_dist / max_dist} 
        
    is_exist, _ = get_intersection_dist(trajectory[0], trajectory[-1], vehicle.trajectory[0], vehicle.trajectory[-1])    
    # print("---------------------------------------------------")
    # print("ego id: ", self.ego.id, " ego_speed: ", get_speed(self.ego)/3.6, " vehicle id: ", vehicle.id, " vehicle_speed: ", get_speed(vehicle)/3.6, " inter_exist: ", is_exist)
    if not is_exist:
      return {"id": vehicle.id, "time_to_collision": max_time / max_time, "dist_to_collision": max_dist / max_dist}
    
    speed = get_speed(vehicle) / 3.6
    buf_span = int(buf_t / self.pred_dt)
    traj_len = len(vehicle.trajectory)
    
    # traj_gap = speed * self.pred_dt
    traj_gap = np.linalg.norm([trajectory[0][0] - trajectory[1][0], trajectory[0][1] - trajectory[1][1]])
    
    dist_to_collision = 0.0    
    for i in range(len(trajectory)-1):
      min_idx, max_idx = max(0, i - buf_span), min(traj_len, i + buf_span)
      # print("---------------------------------------------------")
      # print("ego_locatoin: ", self.ego.get_location(), " vehicle_location: ", vehicle.get_location())
      # print("i: ", i, " min_idx: ", min_idx, " max_idx: ", max_idx)
      # print("ego[i]: ", trajectory[i])
      # print("ego[i+1]: ", trajectory[i+1])
      # print("---------------------------------------------------")
      for k in range(min_idx, max_idx-1):        
        collision_exist, collision_dist = get_intersection_dist(trajectory[i], trajectory[i+1], vehicle.trajectory[k], vehicle.trajectory[k+1])                
        # print("k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist)
        # print("vehicle[k]: ", vehicle.trajectory[k])
        # print("vehicle.trajectory[k+1]: ", vehicle.trajectory[k+1])
        if collision_exist:
          break
      
      # print("---------------------------------------------------")            
      if collision_exist:
        dist_to_collision += collision_dist
        # print("i: ", i, " k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist, " dist_to_collision: ", dist_to_collision)
        break
      else:        
        dist_to_collision += traj_gap
        # dist_to_collsion += np.linalg.norm([trajectory[i][0] - trajectory[i+1][0], trajectory[i][1] - trajectory[i+1][1]])
        # print("i: ", i, " k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist, " dist_to_collision: ", dist_to_collision)
    
    time_to_collision = dist_to_collision / speed    
    time_to_collision = min(time_to_collision, max_time)
    dist_to_collision = min(dist_to_collision, max_dist)
    
    return {"id": vehicle.id, "time_to_collision": time_to_collision / max_time, "dist_to_collision": dist_to_collision / max_dist}
  
  def _apply_random_vehicle_control(self):
    for vehicle in self.vehicles:
      # print("=========================================================")
      # print("vehicle: ", vehicle, " is_alive: ", vehicle.is_alive)
      if vehicle.is_alive:
        control = vehicle.run_step()
        # print("control: ", control)
        vehicle.apply_control(control)
      
  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:      
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          actor.destroy()

  def _clear_all_vehicles(self):
    """Clear all vehicles."""
    if self.vehicles is not None:
      for vehicle in self.vehicles:
        if vehicle.is_alive:
            vehicle.destroy()
