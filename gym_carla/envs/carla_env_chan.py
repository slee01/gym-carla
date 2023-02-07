#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import sys
import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

from agents.navigation.local_planner import LocalPlanner

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self.target_speed = params['target_speed']
    self.other_vehicle_type = params['other_vehicle_filter']
    self.control_all = params['control_all']
    self.topview = params['topview']
    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
      # self.start = [52.1,-4.2, 178.66]
    else:
      self.dests = None

    # action and observation spaces
  
    if self.control_all:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], params['continuous_steering_range'][0]]),
      np.array([params['continuous_accel_range'][1], params['continuous_steering_range'][1]]), dtype=np.float32)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0]]), 
      np.array([params['continuous_accel_range'][1]]), dtype=np.float32)  # acc
    observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
      }
    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
        })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    self.client = carla.Client('localhost', params['port'])
    self.client.set_timeout(10.0)
    self.world = self.client.load_world(params['town'])
    # self.tm_port = 8000
    # self.tm = self.client.get_trafficmanager(self.tm_port)
    # self.tm.set_synchronous_mode(True)
    # self.tm.set_hybrid_physics_mode(True)
    # self.tm.set_hybrid_mode_radius(r=self.obs_range)
    self.ego = None
    # self.lp = None
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)
    self.spectator = self.world.get_spectator()
    
    self.aggressive_vehicles = False
    self.number_of_wheels = [4]
    self.vehicle_list = []
    self.average_vel = None
    self.total_moving_dist = None
    self.total_moving_time = None
    self.prev_pos = None

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    self.collision_sensor = None
    # Initialize the renderer
    self._init_renderer()

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T

  def reset(self):
    # print("RESET ENVIRONMENT")
    # self.number_of_vehicles = 1
    self.total_moving_time = 0.0
    self.total_moving_dist = 0.0
    self.average_vel = 0.0
    self.prev_pos = None
    # Disable sync mode
    self._set_synchronous_mode(False)
    
    if self.ego is not None:
      self.ego.destroy()
    #   carla.command.DestroyActor(self.ego_id)
    # Clear sensor objects  
    if self.collision_sensor is not None:
      self.collision_sensor.stop()
      self.collision_sensor.destroy()
    self.collision_sensor = None
    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['controller.ai.walker', 'walker.*'])
    self._clear_all_vehicles()
    # self.vehicle_list = []
    # Spawn surrounding vehicles
    self.np_random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=self.number_of_wheels):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(self.np_random.choice(self.vehicle_spawn_points), number_of_wheels=self.number_of_wheels):
        count -= 1
    # Spawn pedestrians
    self.np_random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(self.np_random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'random':
        transform = self.np_random.choice(self.vehicle_spawn_points)
      if self.task_mode == 'roundabout':
        self.start=[52.1+self.np_random.uniform(-5,5),-4.2, 178.66] # random
        # self.start = np.add(self.start, [np.random.uniform(-4,4), 0, 0])
        # self.start=[52.1,-4.2, 178.66] # static
        transform = set_carla_transform(self.start)
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        # print("FAILED TO SPAWN EGO VEHICLE")
        ego_spawn_times += 1
        # time.sleep(0.1)
    if self.topview:
      self.spectator.set_transform(carla.Transform(carla.Location(transform.location.x, transform.location.y, transform.location.z+100.0), carla.Rotation(-90.0, 180.0, -90.0)))
    else:
      self.spectator.set_transform(carla.Transform(carla.Location(transform.location.x-10.0*np.cos(transform.rotation.yaw*np.pi/180.0), transform.location.y-10.0*np.sin(transform.rotation.yaw*np.pi/180.0), transform.location.z+5.0), carla.Rotation(pitch=345, yaw=transform.rotation.yaw)))
    
    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
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
    self._set_synchronous_mode(True)
    self._set_autopilots()
    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    self.lp._waypoint_buffer.clear()
    self.lp.set_global_plan(self.routeplanner._waypoint_buffer)
    if self.task_mode == 'random':
      self.dests = [[self.routeplanner._waypoints_queue[-1][0].transform.location.x, self.routeplanner._waypoints_queue[-1][0].transform.location.y, self.routeplanner._waypoints_queue[-1][0].transform.location.z]]
    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)
    self.world.tick()
    # print("RESET FINISHED")

    return self._get_obs()
  
  def step(self, action):
    # Calculate acceleration and steering
    
    acc = action[0]

    # print(len(self.routeplanner._waypoints_queue))
    # print(self.routeplanner._waypoints_queue[-1][0].transform.location)
    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    if self.control_all:
      steer = action[1]
      act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
      self.ego.apply_control(act)
    else:  
      self.lp.set_global_plan(self.routeplanner._waypoint_buffer)
      control = self.lp.run_step()
      control.throttle = throttle
      control.brake = brake
      self.ego.apply_control(control)
    transform = self.ego.get_transform()

    if self.topview:
      self.spectator.set_transform(carla.Transform(carla.Location(transform.location.x, transform.location.y, transform.location.z+100.0), carla.Rotation(-90.0, 180.0, -90.0)))
    else:
      self.spectator.set_transform(carla.Transform(carla.Location(transform.location.x-10.0*np.cos(transform.rotation.yaw*np.pi/180.0), transform.location.y-10.0*np.sin(transform.rotation.yaw*np.pi/180.0), transform.location.z+5.0), carla.Rotation(pitch=345, yaw=transform.rotation.yaw)))
    
    self.world.tick()
    
    current_pos = self.ego.get_location()
    if self.prev_pos is not None:
      self.total_moving_dist += np.sqrt((current_pos.x - self.prev_pos.x)**2 + (current_pos.y - self.prev_pos.y)**2 + (current_pos.z - self.prev_pos.z)**2)
      self.total_moving_time += self.dt
      self.average_vel = self.total_moving_dist/self.total_moving_time
    self.prev_pos = current_pos

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    info = self._terminal()
    done = True in info.values()

    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), done, copy.deepcopy(info))

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
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)
    # self.tm.set_synchronous_mode(synchronous)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    if self.other_vehicle_type is not None:
      if len(self.other_vehicle_type) > 1:
        veh_type = self.np_random.choice(self.other_vehicle_type, 1)
      else:
        veh_type = self.other_vehicle_type
      if veh_type[0] == 'vehicle.bh.crossbike':
        number_of_wheels = [2]
      blueprint = self._create_vehicle_bluepprint(veh_type[0], number_of_wheels=number_of_wheels)
    else:
      blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      self.vehicle_list.append(vehicle.id)
      
      return True
    return False

  def _set_autopilots(self):
    if self.aggressive_vehicles:
      # if bool(random.getrandbits(1)):
      for i in range(len(self.vehicle_list)):
        vehicle = self.world.get_actor(self.vehicle_list[i])
        if vehicle is not None:
          percentage = self.np_random.randint(-50, -20)
          # vehicle.set_autopilot(True, self.tm_port)
          vehicle.set_autopilot(True)
          # self.tm.vehicle_percentage_speed_difference(vehicle, percentage)
          # self.tm.distance_to_leading_vehicle(vehicle, 0.5)
          # if bool(random.getrandbits(1)):
          #   self.tm.ignore_lights_percentage(vehicle, 60)
      # else:
      #   for i in range(len(self.vehicle_list)):
      #     vehicle = self.world.get_actor(self.vehicle_list[i])
      #     if vehicle is not None:
      #       vehicle.set_autopilot(True, self.tm_port)
    else:
      for i in range(len(self.vehicle_list)):
        vehicle = self.world.get_actor(self.vehicle_list[i])
        if vehicle is not None:
          # vehicle.set_autopilot(True, self.tm_port)
          vehicle.set_autopilot(True)

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = self.np_random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + self.np_random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  def _transform_to_ego_local(self, ego_loc, ego_rot, veh_loc):
    x_transform = (veh_loc[0] - ego_loc[0])*np.cos(ego_rot[1]*np.pi/180.0) + (veh_loc[1] - ego_loc[1])*np.sin(ego_rot[1]*np.pi/180.0)
    y_transform = (veh_loc[0] - ego_loc[0])*np.sin(ego_rot[1]*np.pi/180.0) - (veh_loc[1] - ego_loc[1])*np.cos(ego_rot[1]*np.pi/180.0)
    z_transform = veh_loc[2] - ego_loc[2]
    return [x_transform, y_transform, z_transform]

  def _check_overlap(self, ego_trans, veh_poly):
    ego_width = 2.5
    ego_length = 7
    ego_loc = [ego_trans.location.x, ego_trans.location.y, ego_trans.location.z]
    ego_rot = [ego_trans.rotation.pitch, ego_trans.rotation.yaw, ego_trans.rotation.roll]

    for i in range(len(veh_poly)):
      veh_poly_ego_local = self._transform_to_ego_local(ego_loc, ego_rot, [veh_poly[i][0], veh_poly[i][1], 0])
      check_x = self._check_in_range(veh_poly_ego_local[0], -ego_length/2.0, ego_length/2.0)
      check_y = self._check_in_range(veh_poly_ego_local[1], -ego_width/2.0, ego_width/2.0)
      if check_x and check_y:
          return True
    return False

  def _check_in_range(self, query, min, max):
    if query > min and query < max:
      return True
    else:
      return False

  def _try_spawn_ego_vehicle_at(self, transform):
    vehicle = None
    overlap = False
    # Check if ego position overlaps with surrounding vehicles
    for id, poly in self.vehicle_polygons[-1].items():
      if self.ego is not None:
        if id == self.ego.id:
          continue
      overlap = self._check_overlap(transform, poly)
      if not overlap:
        continue
      else:
        break

    if not overlap:
      # if self.ego is None:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
      if vehicle is not None:
        self.ego = vehicle
        self.ego_id = self.ego.id
        # print(self.ego_id)
        self.lp = LocalPlanner(self.ego,  opt_dict={'target_speed' : self.target_speed})
        return True
      # else:
      #   self.ego.set_velocity(carla.Vector3D(x=0, y=0, z=0))
      #   self.ego.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
      #   self.ego.set_transform(transform)
        
        # return True
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

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Roadmap
    if self.pixor:
      roadmap_render_types = ['roadmap']
      if self.display_route:
        roadmap_render_types.append('waypoints')
      self.birdeye_render.render(self.display, roadmap_render_types)
      roadmap = pygame.surfarray.array3d(self.display)
      roadmap = roadmap[0:self.display_size, :, :]
      roadmap = display_to_rgb(roadmap, self.obs_size)
      # Add ego vehicle
      for i in range(self.obs_size):
        for j in range(self.obs_size):
          if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
            roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    lidar = np.zeros([self.obs_size, self.obs_size, 3])
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255

    # Display on pygame
    pygame.display.flip()

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        if actor.id != self.ego.id:
          if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

    obs = {
      'camera':camera.astype(np.uint8),
      'lidar':lidar.astype(np.uint8),
      'birdeye':birdeye.astype(np.uint8),
      'state': state,
    }

    if self.pixor:
      obs.update({
        'roadmap':roadmap.astype(np.uint8),
        'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
        'vh_regr':vh_regr.astype(np.float32),
        'pixor_state': pixor_state,
      })

    return obs

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    r = 0.0

    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1
    r += 300*r_collision
    
    # reward for steering:
    r_steer = -self.ego.get_control().steer**2
    r += 5*r_steer

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1
    r += 1*r_out

    # reward for success
    if self.dests is not None: 
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          r_success = 1
          r += 200*r_success

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)
    r += 1*lspeed_lon
    
    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1
    r += 10*r_fast
    
    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2
    r += 0.2*r_lat
    r -= 0.1

    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)
    terminal_info = {}
    # If collides
    if len(self.collision_hist)>0: 
      # print("Collision occurs!")
      # return True
      terminal_info['collision'] = True
    else:
      terminal_info['collision'] = False

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      # print("Time over!")
      # return True
      terminal_info['timeover'] = True
    else:
      terminal_info['timeover'] = False

    # If at destination
    
    if self.dests is not None: # If at destination
      terminal_info['success'] = False
      if terminal_info['collision'] == False:
        for dest in self.dests:
          if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
            # print("Success!")
            # return True
            terminal_info['success'] = True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      # print("Out of lane!")
      # return True
      terminal_info['outoflane'] = True
    else:
      terminal_info['outoflane'] = False

    terminal_info['mean_velocity'] = self.average_vel
    terminal_info['total_time'] = self.total_moving_time
    terminal_info['total_dist'] = self.total_moving_dist
    
    return terminal_info

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()

  def _clear_all_vehicles(self):
    self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])
    self.vehicle_list = []