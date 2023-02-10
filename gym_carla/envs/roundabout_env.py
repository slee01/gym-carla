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

from gym_carla.envs.carla_env import CarlaEnv
# from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from agents.navigation.safe_agent import SafeAgent


class RoundAboutEnv(CarlaEnv):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    super(RoundAboutEnv, self).__init__(params)
    # parameters
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    else:
      # self.dests = None
      raise NotImplementedError

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    
    # observation_space_dict = {'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)}
    # self.observation_space = spaces.Dict(observation_space_dict)
    self.observation_space = spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
    
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
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
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
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    ############################################################################
    # DEFINE WAYPOINTS AND VEHICLE_FRONT(HAZARD) HERE
    self.waypoints = self.ego.local_planner.get_waypoints(length=50)
    self.vehicle_front =  self.ego.detect_hazard()
    # print("self.waypoints: ", self.waypoints, " length: ", len(self.waypoints))
    # print("self.vehicle_front: ", self.vehicle_front)
    ############################################################################
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
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    self.world.tick()

    # Append actors polygon list
    # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    vehicle_poly_dict = self._get_vehicle_polygons()
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)

    # route planner
    # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    self.waypoints = self.ego.local_planner.get_waypoints(length=50)
    self.vehicle_front =  self.ego.detect_hazard()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }
    
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))
