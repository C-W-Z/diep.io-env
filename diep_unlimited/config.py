from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Constants
    FPS                             = 60
    SCREEN_SIZE                     = 800      # Pixel size of render window
    MAP_SIZE                        = 80
    BORDER_SIZE                     = 5
    MAP_GRID                        = 32        # Size of collision grid
    BASE_MAX_VELOCITY               = 10 / FPS  # 10 grids per second
    BASE_ACC_FRAMES                 = 10        # Acceleration
    BASE_DEC_FRAMES                 = 60        # Deceleration
    POLYGON_BOUNCE_V_SCALE          = 1.2
    POLYGON_BOUNCE_DEC_FRAMES       = 20
    POLYGON_TANK_BOUNCE_V_SCALE     = 0.55
    POLYGON_TANK_BOUNCE_DEC_FRAMES  = 60
    TANK_BOUNCE_V_SCALE             = 0.25
    TANK_TANK_BOUNCE_V_SCALE        = 0.75
    TANK_BOUNCE_DEC_FRAMES          = 40
    TANK_RECOIL_V_SCALE             = 0.075     # Recoil
    TANK_RECOIL_DECAY               = 0.95
    BULLET_COLLIDER_BOUNCE_V_SCALE  = 0.25
    BULLET_BOUNCE_V_SCALE           = 0.25
    BULLET_BOUNCE_DEC_FRAMES        = 5
    INVULNERABLE_FRAMES             = 5
    TANK_LONG_INVULNERABLE_FRAMES   = 15
    SLOW_HP_REGEN_FRAMES            = 30 * FPS  # 30 seconds in 60 fps

    EXP_LIST = [
        0, 0, 4, 13, 28, 50, 78, 113, 157, 211, 275, 350, 437, 538, 655, 787, 938, 1109, 1301, 1516, 1757, 2026, 2325,
        2658, 3026, 3433, 3883, 4379, 4925, 5525, 6184, 6907, 7698, 8537, 9426, 10368, 11367, 12426, 13549, 14730, 16000,
        17337, 18754, 20256, 21849, 23536
    ]
    RESPAWN_LEVEL_LIST = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22]
    FAST_REGEN_LIST = [0.0312, 0.0326, 0.0433, 0.0660, 0.0851, 0.1095, 0.1295, 0.1560]

    N_POLYGON_SCALE                 = 12.5 # n_polygons = floor(n_tanks * N_POLYGON_SCALE)
    POLYGON_SIDE_PROB               = {3: 0.2, 4: 0.7, 5: 0.1}
    POLYGON_SIDE_PROB_LIST          = list(POLYGON_SIDE_PROB.values())
    POLYGON_RADIUS                  = {3: 1.2, 4: 1.1, 5: 1.8}
    POLYGON_COLLISION_RADIUS        = {3: 0.9, 4: 0.9, 5: 1.6}
    POLYGON_HP                      = {3: 30 , 4: 10 , 5: 100}
    POLYGON_BODY_DAMAGE             = {3: 8.0, 4: 8.0, 5: 12.0}
    POLYGON_SCORE                   = {3: 25 , 4: 10 , 5: 130}
    POLYGON_V_SCALE                 = 0.1
    POLYGON_ROTATE_SPEED            = 0.005

    BASE_BULLET_V_SCALE             = 2.0
    MIN_BULLET_V_MULTIPLIER         = 0.5
    BULLET_COLLISION_V_MULTIPLIER   = 0.6
    BULLET_ALIVE_FRAMES             = 3 * FPS

config = Config()
