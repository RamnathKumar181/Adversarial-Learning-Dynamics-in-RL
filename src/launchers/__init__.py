from src.launchers.te_ppo_point import train_te_ppo_pointenv
from src.launchers.ate_ppo_point import train_ate_ppo_pointenv
from src.launchers.te_ppo_mt5 import train_te_ppo_mt5
from src.launchers.ate_ppo_mt5 import train_ate_ppo_mt5
from src.launchers.te_ppo_mt10 import train_te_ppo_mt10
from src.launchers.ate_ppo_mt10 import train_ate_ppo_mt10
from src.launchers.ate_ppo_mt1 import train_ate_ppo_mt1
from src.launchers.te_ppo_mt1 import train_te_ppo_mt1
from src.launchers.ate_ppo_navigation import train_ate_ppo_navigation
from src.launchers.te_ppo_navigation import train_te_ppo_navigation
from src.launchers.ate_ppo_bandit import train_ate_ppo_bandit
from src.launchers.te_ppo_bandit import train_te_ppo_bandit

__all__ = ['train_te_ppo_pointenv', 'train_ate_ppo_pointenv',
           'train_te_ppo_mt5', 'train_ate_ppo_mt5',
           'train_te_ppo_mt1', 'train_ate_ppo_mt1',
           'train_te_ppo_mt10', 'train_ate_ppo_mt10',
           'train_ate_ppo_navigation', 'train_te_ppo_navigation']
