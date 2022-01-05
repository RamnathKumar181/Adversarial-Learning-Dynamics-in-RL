from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerNutAssemblyEnvV2,
    SawyerBasketballEnvV2,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2,
    SawyerNutDisassembleEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerDoorEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2,
    SawyerHandInsertEnvV2,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePullSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerLeverPullEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnvV2,
    SawyerStickPullEnvV2,
    SawyerStickPushEnvV2,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnvV2,
)
from collections import OrderedDict

ALL_V2_ENVIRONMENTS = OrderedDict((
    ('assembly-v2', SawyerNutAssemblyEnvV2),
    ('basketball-v2', SawyerBasketballEnvV2),
    ('bin-picking-v2', SawyerBinPickingEnvV2),
    ('box-close-v2', SawyerBoxCloseEnvV2),
    ('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2),
    ('button-press-topdown-wall-v2', SawyerButtonPressTopdownWallEnvV2),
    ('button-press-v2', SawyerButtonPressEnvV2),
    ('button-press-wall-v2', SawyerButtonPressWallEnvV2),
    ('coffee-button-v2', SawyerCoffeeButtonEnvV2),
    ('coffee-pull-v2', SawyerCoffeePullEnvV2),
    ('coffee-push-v2', SawyerCoffeePushEnvV2),
    ('dial-turn-v2', SawyerDialTurnEnvV2),
    ('disassemble-v2', SawyerNutDisassembleEnvV2),
    ('door-close-v2', SawyerDoorCloseEnvV2),
    ('door-lock-v2', SawyerDoorLockEnvV2),
    ('door-open-v2', SawyerDoorEnvV2),
    ('door-unlock-v2', SawyerDoorUnlockEnvV2),
    ('hand-insert-v2', SawyerHandInsertEnvV2),
    ('drawer-close-v2', SawyerDrawerCloseEnvV2),
    ('drawer-open-v2', SawyerDrawerOpenEnvV2),
    ('faucet-open-v2', SawyerFaucetOpenEnvV2),
    ('faucet-close-v2', SawyerFaucetCloseEnvV2),
    ('hammer-v2', SawyerHammerEnvV2),
    ('handle-press-side-v2', SawyerHandlePressSideEnvV2),
    ('handle-press-v2', SawyerHandlePressEnvV2),
    ('handle-pull-side-v2', SawyerHandlePullSideEnvV2),
    ('handle-pull-v2', SawyerHandlePullEnvV2),
    ('lever-pull-v2', SawyerLeverPullEnvV2),
    ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
    ('pick-place-wall-v2', SawyerPickPlaceWallEnvV2),
    ('pick-out-of-hole-v2', SawyerPickOutOfHoleEnvV2),
    ('reach-v2', SawyerReachEnvV2),
    ('push-back-v2', SawyerPushBackEnvV2),
    ('push-v2', SawyerPushEnvV2),
    ('pick-place-v2', SawyerPickPlaceEnvV2),
    ('plate-slide-v2', SawyerPlateSlideEnvV2),
    ('plate-slide-side-v2', SawyerPlateSlideSideEnvV2),
    ('plate-slide-back-v2', SawyerPlateSlideBackEnvV2),
    ('plate-slide-back-side-v2', SawyerPlateSlideBackSideEnvV2),
    ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
    ('peg-unplug-side-v2', SawyerPegUnplugSideEnvV2),
    ('soccer-v2', SawyerSoccerEnvV2),
    ('stick-push-v2', SawyerStickPushEnvV2),
    ('stick-pull-v2', SawyerStickPullEnvV2),
    ('push-wall-v2', SawyerPushWallEnvV2),
    ('push-v2', SawyerPushEnvV2),
    ('reach-wall-v2', SawyerReachWallEnvV2),
    ('reach-v2', SawyerReachEnvV2),
    ('shelf-place-v2', SawyerShelfPlaceEnvV2),
    ('sweep-into-v2', SawyerSweepIntoGoalEnvV2),
    ('sweep-v2', SawyerSweepEnvV2),
    ('window-open-v2', SawyerWindowOpenEnvV2),
    ('window-close-v2', SawyerWindowCloseEnvV2),
))


# MT5_V1 = OrderedDict(
#     (('push-v2', SawyerPushEnvV2),
#      ('push-wall-v2', SawyerPushWallEnvV2),
#      ('stick-push-v2', SawyerStickPushEnvV2),
#      ('coffee-push-v2', SawyerCoffeePushEnvV2),
#      ('push-back-v2', SawyerPushBackEnvV2),), )
#
# MT5_V1_ARGS_KWARGS = {
#     key: dict(args=[],
#               kwargs={'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
#     for key, _ in MT5_V1.items()
# }

MT5_V2 = OrderedDict(
    (('push-v2', SawyerPushEnvV2),
     ('window-open-v2', SawyerWindowOpenEnvV2),
     ('window-close-v2', SawyerWindowCloseEnvV2),
     ('drawer-close-v2', SawyerDrawerCloseEnvV2),
     ('drawer-open-v2', SawyerDrawerOpenEnvV2),
     ('push-wall-v2', SawyerPushWallEnvV2),
     ('stick-push-v2', SawyerStickPushEnvV2),
     ('coffee-push-v2', SawyerCoffeePushEnvV2),
     ('push-back-v2', SawyerPushBackEnvV2),
     ('handle-press-v2', SawyerHandlePressEnvV2)), )

MT5_V2_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT5_V2.items()
}


MT5_V1 = OrderedDict(
    (('push-v2', SawyerPushEnvV2),
     ('window-open-v2', SawyerWindowOpenEnvV2),
     ('window-close-v2', SawyerWindowCloseEnvV2),
     ('drawer-close-v2', SawyerDrawerCloseEnvV2),
     ('drawer-open-v2', SawyerDrawerOpenEnvV2),), )

MT5_V1_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT5_V1.items()
}
