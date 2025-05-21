import numpy as np
from unit import Unit, UnitType
from config import config as cfg

class Polygon(Unit):
    def __init__(self, x, y,side):
        self.side = side
        assert side in cfg.POLYGON_RADIUS, f"Invalid polygon side count: {side}"

        super().__init__(
            unit_type=UnitType.Polygon,
            x=x,
            y=y,
            max_hp = cfg.POLYGON_HP[side],
            body_damage=cfg.POLYGON_BODY_DAMAGE[side],
            radius=cfg.POLYGON_RADIUS[side],
            score=cfg.POLYGON_SCORE[side],
        )

        self.angle = np.random.uniform(-np.pi, np.pi)
