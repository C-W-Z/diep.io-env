import numpy as np
from unit import Unit, UnitType
from config import config as cfg

class Bullet(Unit):
    def __init__(self, x, y, max_hp, bullet_damage, radius, rx, ry):

        super().__init__(
            unit_type=UnitType.Polygon,
            x=x,
            y=y,
            max_hp = max_hp,
            body_damage=bullet_damage,
            radius=radius,
            score=0,
        )

        self.rx = rx
        self.ry = ry
