import numpy as np
from unit import Unit, UnitType
from config import config as cfg
from tank import Tank

class Bullet(Unit):
    def __init__(
        self,
        x,
        y,
        max_hp,
        bullet_damage,
        radius,
        tank: Tank,
        rx,
        ry,
        v_scale,
    ):

        super().__init__(
            unit_type=UnitType.Bullet,
            x=x,
            y=y,
            max_hp = max_hp,
            body_damage=bullet_damage,
            radius=radius,
            score=0,
        )

        self.tank = tank
        self.rx = rx
        self.ry = ry
        self.v_scale = v_scale

        self.slow_health_regen = 0
        self.fast_health_regen = 0

    def deal_damage(self, collider: "Unit", self_hp_before_hit = None):
        super(Bullet, self).deal_damage(collider, self_hp_before_hit)
        if self.score > 0:
            self.tank.score += self.score
            self.score = 0

    def update(self, colhash):
        """
        Move the bullet, update its position in the collision hash,
        and check boundaries. Return False if the bullet should be removed.
        """
        # save old position for collision-hash update
        old_x, old_y = self.x, self.y

        # linear motion along facing direction
        self.x += self.rx * self.v_scale
        self.y += self.ry * self.v_scale

        # update spatial hash grid cell
        colhash.update(old_x, old_y, self.x, self.y, self.id)

        # remove bullet if it goes out of map bounds
        min_coord = self.radius
        max_coord = cfg.MAP_SIZE - self.radius
        if not (min_coord <= self.x <= max_coord and min_coord <= self.y <= max_coord):
            return False

        return True
