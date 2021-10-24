from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='src.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='src.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v2',
    entry_point='src.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'src.envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v2',
    entry_point='src.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'src.envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v1',
    entry_point='src.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'src.envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v2',
    entry_point='src.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'src.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='src.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'src.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='src.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
