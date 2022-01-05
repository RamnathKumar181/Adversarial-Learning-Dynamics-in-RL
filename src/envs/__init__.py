from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='src.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
