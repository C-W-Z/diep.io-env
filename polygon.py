import numpy as np
from unit import Unit, UnitType
from config import config as cfg

class Polygon(Unit):
    def __init__(self, x, y, side, new_id=True):
        self.side = side
        assert side in cfg.POLYGON_COLLISION_RADIUS, f"Invalid polygon side count: {side}"

        super().__init__(
            unit_type=UnitType.Polygon,
            x=x,
            y=y,
            max_hp = cfg.POLYGON_HP[side],
            body_damage=cfg.POLYGON_BODY_DAMAGE[side],
            radius=cfg.POLYGON_COLLISION_RADIUS[side],
            score=cfg.POLYGON_SCORE[side],
            new_id=new_id,
        )

        self.angle = np.random.uniform(-np.pi, np.pi)
        self.rotate_dir = 1 if np.random.rand() < 0.5 else -1
        self.v_scale = cfg.POLYGON_V_SCALE

    def update_direction(self):
        self.angle += self.rotate_dir * cfg.POLYGON_ROTATE_SPEED
        if self.angle > 2 * np.pi:
            self.angle -= 2 * np.pi
        elif self.angle < -2 * np.pi:
            self.angle += 2 * np.pi

        self.rx = np.cos(self.angle)
        self.ry = np.sin(self.angle)
