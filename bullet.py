import numpy as np
from unit import Unit, UnitType
from config import config as cfg
from tank import Tank
from typing import Optional

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
        new_id=True
    ):

        super().__init__(
            unit_type=UnitType.Bullet,
            x=x,
            y=y,
            max_hp = max_hp,
            body_damage=bullet_damage,
            radius=radius,
            score=0,
            new_id=new_id
        )

        self.tank = tank
        self.rx = rx
        self.ry = ry
        self.v_scale = v_scale
        self.move_frame = 0

        self.slow_health_regen = 0
        self.fast_health_regen = 0

    def add_score(self, score: int):
        self.tank.add_score(score)

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

            self.total_vx = max(self.rx * min_v, self.vx * cfg.BULLET_COLLISION_V_MULTIPLIER) + collision_vx
            self.total_vy = max(self.ry * min_v, self.vy * cfg.BULLET_COLLISION_V_MULTIPLIER) + collision_vy
        else:
            self.collision_frame = 0
            self.collision_vx = 0.0
            self.collision_vy = 0.0

            self.total_vx = self.vx
            self.total_vy = self.vy

        self.x += self.total_vx
        self.y += self.total_vy

        # remove bullet if it goes out of map bounds
        if not (-self.radius <= self.x <= cfg.MAP_SIZE + self.radius and -self.radius <= self.y <= cfg.MAP_SIZE + self.radius):
            self.hp = 0.0

class BulletPool:
    def __init__(self, max_bullets: int = 18):
        self.max_bullets = max_bullets
        self.bullets: list[Bullet] = []

        # Pre-create bullet instances with unique IDs
        for _ in range(max_bullets):
            bullet = Bullet(
                x=0.0,
                y=0.0,
                max_hp=0,
                bullet_damage=0,
                radius=0.5,
                tank=None,
                rx=0.0,
                ry=0.0,
                v_scale=1.0,
            )
            self.bullets.append(bullet)

    def get_new_bullet(self, tank: Tank) -> Optional[Bullet]:
        """Get an inactive bullet from the pool if available."""
        for bullet in self.bullets:
            if bullet.alive:
                continue
            # Reset bullet properties
            bx = tank.x + tank.radius * tank.rx * 2
            by = tank.y + tank.radius * -tank.ry * 2
            bullet.__init__(
                x=bx,
                y=by,
                max_hp = tank.bullet_max_hp,
                bullet_damage = tank.bullet_damage,
                radius = tank.bullet_radius,
                tank = tank,
                rx = tank.rx,
                ry = -tank.ry,
                v_scale = tank.bullet_v_scale,
                new_id=False
            )
            return bullet
        return None
