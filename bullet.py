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
        self.move_frame = 0

        self.slow_health_regen = 0
        self.fast_health_regen = 0

    def deal_damage(self, collider: "Unit", self_hp_before_hit = None):
        super(Bullet, self).deal_damage(collider, self_hp_before_hit)
        if self.score > 0:
            self.tank.score += self.score
            self.score = 0

    def update_counter(self):
        if self.invulberable_frame > 0:
            self.invulberable_frame -= 1
        self.move_frame += 1
        if self.move_frame >= cfg.BULLET_ALIVE_FRAMES:
            self.hp = 0.0

    def move(self):
        if not self.alive:
            return

        magnitude = np.hypot(self.rx, self.ry)
        self.rx, self.ry = self.rx / magnitude, self.ry / magnitude

        # Handle normal motion
        max_v = cfg.BASE_MAX_VELOCITY * self.v_scale
        min_v = max_v * cfg.MIN_BULLET_V_MULTIPLIER

        if self.move_frame <= cfg.BASE_ACC_FRAMES:
            self.ax = self.rx * max_v / cfg.BASE_ACC_FRAMES
            self.ay = self.ry * max_v / cfg.BASE_ACC_FRAMES
        else:
            speed = np.hypot(self.vx, self.vy)

            if speed > min_v:
                dx, dy = -self.vx / (speed), -self.vy / (speed)
                self.ax = dx * (max_v - min_v) / cfg.BASE_DEC_FRAMES
                self.ay = dy * (max_v - min_v) / cfg.BASE_DEC_FRAMES
            else:
                self.ax = 0.0
                self.ay = 0.0
                self.vx = self.rx * min_v
                self.vy = self.ry * min_v

        self.vx += self.ax
        self.vy += self.ay
        normal_speed = np.hypot(self.vx, self.vy)
        if normal_speed > max_v:
            self.vx = self.vx * max_v / normal_speed
            self.vy = self.vy * max_v / normal_speed

        # Handle collision motion
        collision_vx, collision_vy = 0.0, 0.0
        if self.collision_frame > 0:
            factor = self.collision_frame / self.max_collision_frame
            collision_vx = self.collision_vx * factor
            collision_vy = self.collision_vy * factor
            self.collision_frame -= 1

            final_vx = max(self.rx * min_v, self.vx * cfg.BULLET_COLLISION_V_MULTIPLIER) + collision_vx
            final_vy = max(self.ry * min_v, self.vy * cfg.BULLET_COLLISION_V_MULTIPLIER) + collision_vy
        else:
            self.collision_frame = 0
            self.collision_vx = 0.0
            self.collision_vy = 0.0

            final_vx = self.vx
            final_vy = self.vy

        self.x += final_vx
        self.y += final_vy

        # remove bullet if it goes out of map bounds
        if not (-self.radius <= self.x <= cfg.MAP_SIZE + self.radius and -self.radius <= self.y <= cfg.MAP_SIZE + self.radius):
            self.hp = 0.0
