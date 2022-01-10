from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='src.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )
# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='src.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

# Minigrid
# ----------------------------------------

register(
    id='MiniGrid-MultiRoom-N2-S4-v0',
    entry_point='src.envs.minigrid:MultiRoomEnvN2S4'
)

register(
    id='MiniGrid-MultiRoom-N4-S5-v0',
    entry_point='src.envs.minigrid:MultiRoomEnvN4S5'
)

register(
    id='MiniGrid-MultiRoom-N6-v0',
    entry_point='src.envs.minigrid:MultiRoomEnvN6'
)
