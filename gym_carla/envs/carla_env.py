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
    # self.pred_dist = params['pred_dist']
    self.pred_dt = self.dt * 10.0
    self.pred_time, self.pred_dist = 5.0, 80.0
    self.spawn_range = 60.0
    
    self.dests = None
    self.ego_init, self.ego_dest = None, None
    self.collision_infos = None
    self.discrete, self.discrete_act = None, None
    self.n_acc, self.n_steer = None, None
    self.action_space = None
    self.observation_space = None
    self.action_types = None

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

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'random':
        transform = random.choice(self.vehicle_spawn_points)
      if self.task_mode == 'roundabout':
        # self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        self.start=[62.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        transform = set_carla_transform(self.start)
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)
        
    self._update_ego_vehicle_path()
    # self._update_vehicle_waypoints_and_trajectory()
    
    # print("ego_location: ", self.ego.get_location())
    # for i, vehicle in enumerate(self.vehicles):
    #   print("i: ", i, " vehicle location: ", vehicle.get_location())

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        # if np.linalg.norm([spawn_point.location.x - 52.1, spawn_point.location.y - 4.2]) < 10.0:
          # continue
        if np.linalg.norm([spawn_point.location.x, spawn_point.location.y]) < self.spawn_range:
            continue          
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      spawn_point = random.choice(self.vehicle_spawn_points)
      if np.linalg.norm([spawn_point.location.x, spawn_point.location.y]) < self.spawn_range:
          continue
      if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
        count -= 1

    self._update_random_vehicle_paths()

    # Get actors polygon list
    self.vehicle_polygons = []
    # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    vehicle_poly_dict = self._get_vehicle_polygons()
    self.vehicle_polygons.append(vehicle_poly_dict)

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
    self._update_ego_vehicle_waypoints_and_trajectories()
    self._update_random_vehicle_waypoints_and_trajectories()
    
    # print("Detect Ego Vehicle Hazard")
    self.vehicle_front =  self.ego.detect_hazard()
    # print("self.waypoints: ", self.waypoints, " length: ", len(self.waypoints))
    # print("self.vehicle_front: ", self.vehicle_front)
    ############################################################################
    
    self._set_dist_to_collisions()
    self._update_ego_vehicle_desired_speeds()
    self._set_time_to_collisions()
    # self.ego.waypoints = self.ego.cand_wpts[0]['waypoints']

    # print("Complete Reset")
    return self._get_obs(action=0)

  def step(self, action):
    # Apply control
    # action = 0
    act = self.ego.local_planner.get_control(
      waypoints=self.ego.cand_wpts[action],
      target_speed=self.ego.desired_speeds[action], 
      collision_info=self.collision_infos[action])
    self.ego.apply_control(act)
    self._apply_random_vehicle_control()

    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_vehicle_polygons()
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)

    self._update_ego_vehicle_waypoints_and_trajectories()
    self._update_random_vehicle_waypoints_and_trajectories()
    self.vehicle_front =  self.ego.detect_hazard()

    self._set_dist_to_collisions()
    self._update_ego_vehicle_desired_speeds()
    self._set_time_to_collisions()
    # self.ego.waypoints = self.ego.cand_wpts[action]['waypoints']

    print("*********************************************************")
    print("location: ", self.ego.get_location(), "desired_speed: ", self.ego.desired_speeds[action], " current_speed: ", get_speed(self.ego))
    print("control: ", act)
    # print("waypoints: ", self.ego.cand_wpts[0]['waypoints'][:5])

    # state information
    info = {
      'waypoints': self.ego.cand_wpts[0]['waypoints'],
      'vehicle_front': self.vehicle_front
    }
    
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    # return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))
    return (self._get_obs(action=action), self._get_reward(action=action), self._terminal(action=action), copy.deepcopy(info))

  def _get_obs(self, action=None):
    raise NotImplementedError
  
  def _get_reward(self, action=None):
    raise NotImplementedError
  
  def _terminal(self, action=None):
    raise NotImplementedError

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
      # vehicle = SafeAgent(vehicle, behavior="normal", dt=self.pred_dt, target_speed=20.0)
      vehicle = SafeAgent(vehicle, behavior="cautious", dt=self.pred_dt, target_speed=20.0)
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
    # for idx, poly in self.vehicle_polygons[-1].items():
    #   poly_center = np.mean(poly, axis=0)
    #   ego_center = np.array([transform.location.x, transform.location.y])
    #   dis = np.linalg.norm(poly_center - ego_center)
    #   if dis > 8:
    #     continue
    #   else:
    #     overlap = True
    #     break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=SafeAgent(vehicle, behavior="normal", dt=self.pred_dt, target_speed=20.0)
      self.ego_init = self.ego.get_location()
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

  def _update_ego_vehicle_path(self):
    if self.ego is not None and self.dests is not None: 
      _dest = random.choice(self.dests)
      self.ego_dest = _dest
      self.ego.update_destination(end_location=carla.Location(_dest[0], _dest[1], _dest[2]))
    else:
      raise NotImplementedError

  def _update_random_vehicle_paths(self):
    if self.dests is not None: # If at destination
      for vehicle in self.vehicles:
        # print("self.dests: ", self.dests)  # [[x,y,z], [x,y,z], [x,y,z], ..., [x,y,z]]
        _dest = random.choice(self.dests)
        # vehicle.update_destination(random.choice(self.dests))
        vehicle.update_destination(end_location=carla.Location(_dest[0], _dest[1], _dest[2]))
        # print("vehicle.id: ", vehicle._vehicle.id, " location: ", vehicle.get_location(), "destinaiton: ", _dest)
        # print("vehicle.local_planner.get_waypoints(): ", vehicle.local_planner.get_waypoints())
        # print("vehicle.local_planner._waypoint_buffer: ", vehicle.local_planner._waypoint_buffer)
    else:
      raise NotImplementedError
  
  def _update_ego_vehicle_waypoints_and_trajectories(self):
    self.ego.set_candidate_waypoints(self.action_types)
    self.ego.set_candidate_trajectories()

  def _update_random_vehicle_waypoints_and_trajectories(self):
    # self.ego.update_trajectory(max_t=max_t)
    for vehicle in self.vehicles:
      # vehicle.pred_trajs = vehicle.get_trajectory(max_t=self.pred_time)
      vehicle.set_predicted_waypoints()
      vehicle.set_predicted_trajectories(max_t=self.pred_time)
      
  def _set_dist_to_collisions(self):
    if self.ego.cand_trajs is None:
      raise NotImplementedError

    self.collision_infos = []
    for cand_traj in self.ego.cand_trajs:
      collisions = []
      for vehicle in self.vehicles:
        if vehicle.is_alive:
          collision = self._get_dist_to_collision(cand_traj, vehicle, max_time=self.pred_time)
          collisions.append(collision)
          
      collisions = sorted(collisions, key=lambda d: d['dist_to_collision'], reverse=False)
      self.collision_infos.append(collisions)

  def _set_time_to_collisions(self):    
    if not self.collision_infos:
      raise NotImplementedError
    
    assert len(self.ego.desired_speeds) == len(self.collision_infos), "desired_speeds of ego vehicle and collision_infos should have same length."
    
    for i, collision_info in enumerate(self.collision_infos):
      for j in range(len(collision_info)):
        if collision_info[j]['dist_to_collision'] == self.pred_dist:
          time_to_collision = self.pred_time
        else:
          time_to_collision = collision_info[j]['dist_to_collision'] / (self.ego.desired_speeds[i] / 3.6)
        collision_info[j]['time_to_collision'] = time_to_collision

    # if self.ego.cand_trajs is None:
    #   raise NotImplementedError
    
    # self.collision_infos = []
    # for cand_traj in self.ego.cand_trajs:
    #   collisions = []
    #   for vehicle in self.vehicles:
    #     if vehicle.is_alive:
    #       collision = self._get_time_and_dist_to_collision(cand_traj, vehicle, max_time=self.pred_time)
    #       collisions.append(collision)
          
    #   collisions = sorted(collisions, key=lambda d: d['time_to_collision'], reverse=False)
    #   self.collision_infos.append(collisions)

  def _get_dist_to_collision(self, cand_traj, vehicle, buf_t = 2.0, max_time=5.0, max_dist=80.0):
    # print("len(cand_traj): ", len(cand_traj), " len(vehicle.pred_trajs): ", len(vehicle.pred_trajs), " len(vehicle.pred_trjas[0]): ", len(vehicle.pred_trajs[0]))
    # print("vehicle.pred_trajs: ", vehicle.pred_trajs, " ", len(vehicle.pred_trajs[0]))
    if len(cand_traj) < 2 or len(vehicle.pred_trajs[0]) < 2:
      return {"id": vehicle.id, "collision": False, "speed": get_speed(vehicle), "dist_to_collision": max_dist}
    
    is_exist, _ = get_intersection_dist(cand_traj[0], cand_traj[-1], vehicle.pred_trajs[0][0], vehicle.pred_trajs[0][-1])    
    if is_exist:
      print("========================================================")
      print("cand_traj[0]: ", cand_traj[0])
      print("cand_traj[-1]: ", cand_traj[-1])
      print("pred_traj[0][0]: ", vehicle.pred_trajs[0][0])
      print("pred_traj[0][-1]: ", vehicle.pred_trajs[0][-1])
      print("is_exist: ", is_exist)
      print("========================================================")
    # else:
      # print("========================================================")
      # print("")
      # print("========================================================")
       
    if not is_exist:
      return {"id": vehicle.id, "collision": False, "speed": get_speed(vehicle), "dist_to_collision": max_dist}
    
    buf_span = int(buf_t / self.pred_dt)
    traj_len = len(vehicle.pred_trajs[0])    
    traj_gap = np.linalg.norm([cand_traj[0][0] - cand_traj[1][0], cand_traj[0][1] - cand_traj[1][1]])
    dist_to_collision = 0.0    
    for i in range(len(cand_traj)-1):
      min_idx, max_idx = max(0, i - buf_span), min(traj_len, i + buf_span + 1)
      print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("ego_traj[i]: ", i, " ", cand_traj[i])
      print("ego_traj[i+1]: ", i+1, " ", cand_traj[i+1])
      print("min_idx: ", min_idx, " max_idx: ", max_idx)
      for k in range(min_idx, max_idx-1):        
        print("##########################################")
        collision_exist, collision_dist = get_intersection_dist(cand_traj[i], cand_traj[i+1], vehicle.pred_trajs[0][k], vehicle.pred_trajs[0][k+1])                
        print("k: ", k, " collision_exist, ", collision_exist, " collision_dist: ", collision_dist)
        print("vehicle_traj[k]: ", vehicle.pred_trajs[0][k])
        print("vehicle_traj[k+1]: ", vehicle.pred_trajs[0][k+1])
        if collision_exist:
          break
            
      if collision_exist:
        dist_to_collision += collision_dist
        break
      else:        
        dist_to_collision += traj_gap
    
    dist_to_collision = min(dist_to_collision, max_dist)    
    return {"id": vehicle.id, "collision": False, "speed": get_speed(vehicle), "dist_to_collision": dist_to_collision}
    
  def _update_ego_vehicle_desired_speeds(self):
    assert len(self.ego.cand_trajs) == len(self.collision_infos), "candidate trajectories and collision_infos of ego vehicle should be equal."
    self.ego.desired_speeds = []

    for i, collision_info in enumerate(self.collision_infos):
      # print("****************************************************************************************")      
      # print("i: ", i, " collision_info: ", collision_info)
      desired_speed = self.ego.get_target_speed(front_dist=collision_info[0]['dist_to_collision'], front_speed=collision_info[0]['speed'], waypoint_type=self.action_types[i])
      self.ego.desired_speeds.append(desired_speed)

  # def _get_time_and_dist_to_collision(self, cand_traj, vehicle, buf_t = 2.0, max_time=5.0, max_dist=80.0):
  #   if len(cand_traj) < 2 or len(vehicle.pred_traj) < 2:
  #     return {"id": vehicle.id, "time_to_collision": max_time / max_time, "dist_to_collision": max_dist / max_dist} 
        
  #   is_exist, _ = get_intersection_dist(cand_traj[0], cand_traj[-1], vehicle.pred_traj[0], vehicle.pred_traj[-1])    
  #   # print("---------------------------------------------------")
  #   # print("ego id: ", self.ego.id, " ego_speed: ", get_speed(self.ego)/3.6, " vehicle id: ", vehicle.id, " vehicle_speed: ", get_speed(vehicle)/3.6, " inter_exist: ", is_exist)
  #   if not is_exist:
  #     return {"id": vehicle.id, "time_to_collision": max_time / max_time, "dist_to_collision": max_dist / max_dist}
    
  #   speed = get_speed(vehicle) / 3.6
  #   buf_span = int(buf_t / self.pred_dt)
  #   traj_len = len(vehicle.pred_traj)
    
  #   # traj_gap = speed * self.pred_dt
  #   traj_gap = np.linalg.norm([cand_traj[0][0] - cand_traj[1][0], cand_traj[0][1] - cand_traj[1][1]])
    
  #   dist_to_collision = 0.0    
  #   for i in range(len(cand_traj)-1):
  #     min_idx, max_idx = max(0, i - buf_span), min(traj_len, i + buf_span)
  #     # print("---------------------------------------------------")
  #     # print("ego_locatoin: ", self.ego.get_location(), " vehicle_location: ", vehicle.get_location())
  #     # print("i: ", i, " min_idx: ", min_idx, " max_idx: ", max_idx)
  #     # print("ego[i]: ", cand_traj[i])
  #     # print("ego[i+1]: ", cand_traj[i+1])
  #     # print("---------------------------------------------------")
  #     for k in range(min_idx, max_idx-1):        
  #       collision_exist, collision_dist = get_intersection_dist(cand_traj[i], cand_traj[i+1], vehicle.pred_traj[k], vehicle.pred_traj[k+1])                
  #       # print("k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist)
  #       # print("vehicle[k]: ", vehicle.pred_traj[k])
  #       # print("vehicle.pred_traj[k+1]: ", vehicle.pred_traj[k+1])
  #       if collision_exist:
  #         break
      
  #     # print("---------------------------------------------------")            
  #     if collision_exist:
  #       dist_to_collision += collision_dist
  #       # print("i: ", i, " k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist, " dist_to_collision: ", dist_to_collision)
  #       break
  #     else:        
  #       dist_to_collision += traj_gap
  #       # dist_to_collsion += np.linalg.norm([cand_traj[i][0] - cand_traj[i+1][0], cand_traj[i][1] - cand_traj[i+1][1]])
  #       # print("i: ", i, " k: ", k, " collision_exist: ", collision_exist, " collision_dist: ", collision_dist, " dist_to_collision: ", dist_to_collision)
    
  #   time_to_collision = dist_to_collision / speed    
  #   time_to_collision = min(time_to_collision, max_time)
  #   dist_to_collision = min(dist_to_collision, max_dist)
    
  #   return {"id": vehicle.id, "time_to_collision": time_to_collision / max_time, "dist_to_collision": dist_to_collision / max_dist}

  def _apply_random_vehicle_control(self):
    for vehicle in self.vehicles:
      # print("=========================================================")
      # print("vehicle: ", vehicle, " location: ", vehicle.get_location(), " speed: ", get_speed(vehicle))
      if vehicle.is_alive:
        control = vehicle.run_step()
        # print("control: ", control)
        # print("waypoints: ", vehicle.pred_wpts[0][:5])
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
