from gym.envs.registration import register

register(
    id='roundabout-v0',
    entry_point='gym_carla.envs:RoundAboutEnv',
)

register(
    id='lanechange-v0',
    entry_point='gym_carla.envs:LaneChangeEnv',
)

register(
    id='intersection-v0',
    entry_point='gym_carla.envs:IntersectionEnv',
)