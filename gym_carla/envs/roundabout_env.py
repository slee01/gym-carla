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
    if self.town != "Town03":
      raise NotImplementedError
    
    # Destination
    self.start=[62.1,-4.2, 178.66]
    self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    self.vehicle_spawn_points = self._get_near_spawn_points(loc=carla.Location(x=0.0, y=0.0, z=0.0))

    # action and observation spaces
    # self.discrete = params['discrete']
    # self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    # self.n_acc = len(self.discrete_act[0])
    # self.n_steer = len(self.discrete_act[1])

    # if self.discrete:
    #   self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    # else:
    #   self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
    #   params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
    #   params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    #     
    self.action_types = ["GO", "STOP"]
    self.number_of_detections = 1
    self.action_space = spaces.Discrete(len(self.action_types))

    # observation_space_dict = {'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)}
    # self.observation_space = spaces.Dict(observation_space_dict)
    self.observation_space = spaces.Box(
      np.array([-2, -1, -5, 0, 0.0, 0.0, 0.0, 0.0]), 
      np.array([2, 1, 30, 1, 1.0, 1.0, 1.0, 1.0]), 
      dtype=np.float32)

  def _get_obs(self, action=None):
    """Get the observations."""

    # current_waypoint = self.world.get_map().get_waypoint(self.ego.get_location())
    # next_waypoint = current_waypoint.next(100.0)
    # print("next_waypoint = current_waypoint.next(100.0) ", len(next_waypoint))
    # print(next_waypoint)
    # for nwp in next_waypoint:
    #   print("waypoint: ", nwp)      
    #   self.world.debug.draw_string(
    #   nwp.transform.location, 'o', 
    #   draw_shadow=False, 
    #   color=carla.Color(r=255, g=255,b=255), 
    #   life_time=600.0, 
    #   persistent_lines=True)
    
    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi 
    lateral_dis, w = get_preview_lane_dis(self.ego.cand_wpts[action]['waypoints'], ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
      
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    time_to_collisions = np.ones(len(self.action_types) * self.number_of_detections)
    dist_to_collisions = np.ones(len(self.action_types) * self.number_of_detections)
    if self.collision_infos:
      for i in range(self.number_of_detections):
        for j in range(len(self.action_types)):
          index = i * len(self.action_types) + j
          time_to_collisions[index] = self.collision_infos[i][j]['time_to_collision'] / self.pred_time
          dist_to_collisions[index] = self.collision_infos[i][j]['dist_to_collision'] / self.pred_dist
      
    # print("state: ", state)
    # print("time_to_collisions: ", time_to_collisions)
    # print("dist_to_collisions: ", dist_to_collisions)
    state = np.concatenate((state, time_to_collisions, dist_to_collisions))
    # print("concat_state: ", state)
    
    # obs = {'state': state,}
    # return obs
    return state

  def _get_reward(self, action=None):
    """Calculate the step reward."""
    # reward for speed tracking
    # v = self.ego.get_velocity()
    # speed = np.sqrt(v.x**2 + v.y**2)
    speed = get_speed(self.ego)
    # r_speed = -abs(speed - self.desired_speed)
    # r_speed = min(1.0, speed / self.desired_speed)
    r_speed = min(1.0, speed / self.ego._behavior.max_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0 or self.collision_infos[action][0]['time_to_collision'] <= 1.0:
      r_collision = -1

    r = 10*r_collision + 1*r_speed

    # print("speed: ", speed, " self.collision_hist: ", self.collision_hist, " self.collision_infos[action][0]['time_to_collision']: ", self.collision_infos[action][0]['time_to_collision'])
    return r

  def _terminal(self, action=None):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # print("if len(self.collision_hist)>0: ", len(self.collision_hist)>0)
    # print("if self.time_step>=self.max_time_episode: ", self.time_step, " ", self.max_time_episode)
    # print("if np.sqrt((ego_x-self.ego_dest[0])**2+(ego_y-self.ego_dest[1])**2)<10.0: ", np.sqrt((ego_x-self.ego_dest[0])**2+(ego_y-self.ego_dest[1])**2))
    # print("get_lane_dis(self.ego.cand_wpts[action]['waypoints'], ego_x, ego_y): ", get_lane_dis(self.ego.cand_wpts[action]['waypoints'], ego_x, ego_y))

    # # If collides
    # if len(self.collision_hist)>0: 
    #   return True
    # If reach maximum timestep
    if self.time_step>=self.max_time_episode:
      return True

    # # If at destination
    # if self.dests is not None: # If at destination
    #   for dest in self.dests:
    #     if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
    #       return True    

    if np.sqrt((ego_x-self.ego_dest[0])**2+(ego_y-self.ego_dest[1])**2)<4.0:
      return True

    # # If out of lane
    # dis, _ = get_lane_dis(self.ego.cand_wpts[action]['waypoints'], ego_x, ego_y)
    # if abs(dis) > self.out_lane_thres:
    #   return True
    return False
