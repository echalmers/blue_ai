from gymnasium.envs.registration import register

register(
    id='blue_ai_envs/TransientGoals',
    entry_point='blue_ai_envs.envs:TransientGoals'
)