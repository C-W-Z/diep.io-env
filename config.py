from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Constants
    FPS = 60
    SCREEN_SIZE = 1000  # Pixel size of render window
    MAP_SIZE = 80
    BOARDER_SIZE = 5
    BASE_MAX_VELOCITY = 0.5
    BASE_ACCELLERATION_FRAMES = 20
    BASE_DECCELLERATION_FRAMES = 30
    COLLISION_BOUNCE_V_SCALE = 0.75
    COLLISION_BOUNCE_DECCELLERATE_FRAMES = 40
    INVULNERABLE_FRAMES = 5
    SLOW_HP_REGEN_FRAMES = 30 * FPS  # 30 seconds in 60 fps

config = Config()
