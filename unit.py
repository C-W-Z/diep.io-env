import numpy as np
from enum import Enum
from config import config as cfg
import itertools

class UnitType(Enum):
    Tank    = 0
    Polygon = 1
    Bullet  = 2

class Unit:
    id_iter = itertools.count()

    @classmethod
    def reset_id_iter(self):
        Unit.id_iter = itertools.count()

    def __init__(
        self,
        unit_type,
        x,
        y,
        max_hp=50.0,
        body_damage=8.0,
        radius=1.0,
        score=0,
        new_id=True,
    ):
        if new_id:
            self.id          = next(Unit.id_iter)

        self.type        = unit_type
        self.score       = score
        self.max_hp      = max_hp
        self.hp          = max_hp
        self.radius      = radius
        self.v_scale     = 1.0
        self.x , self.y  = x  , y
        self.rx, self.ry = 1.0, 0.0 # facing direction
        self.vx, self.vy = 0.0, 0.0
        self.ax, self.ay = 0.0, 0.0
        self.collision_vx, self.collision_vy = 0.0, 0.0
        self.recoil_vx, self.recoil_vy = 0.0, 0.0
        self.max_collision_frame = 0
        self.collision_frame     = 0
        self.invulberable_frame  = 0
        self.hp_regen_frame      = 0
        # Stats Properties
        self.slow_health_regen = 0.03 / 30 / cfg.FPS # per frame, == 3% per 30 second
        self.fast_health_regen = 0.0312 / cfg.FPS # per frame, == 3.12% per second
        self.body_damage       = body_damage

    @property
    def alive(self):
        return self.hp > 0

    def add_score(self, score):
        # TODO: Bullet, Tank should overwrite
        self.score += score

    def deal_damage(self, collider: "Unit", self_hp_before_hit = None):
        if self_hp_before_hit == None:
            self_hp_before_hit = self.hp
        body_dmg = collider.body_damage * (0.25 if self.type == UnitType.Bullet else 1.0)
        damage_scale = 1.0
        if collider.type == UnitType.Bullet:
            damage_scale = 0.25
        elif collider.type == UnitType.Tank and self.type == UnitType.Tank:
            damage_scale = 1.5
        if self_hp_before_hit >= body_dmg:
            collider.hp -= damage_scale * self.body_damage
        else:
            collider.hp -= damage_scale * self.body_damage * self_hp_before_hit / body_dmg
        if collider.hp < 0:
            collider.hp = 0.0
            self.add_score(min(collider.score, cfg.EXP_LIST[-1]))
        else:
            collider.hp_regen_frame = cfg.SLOW_HP_REGEN_FRAMES
            collider.invulberable_frame = cfg.INVULNERABLE_FRAMES

            if collider.type == UnitType.Tank and collider.same_collider_counter == 2 and collider.last_collider_id == self.id:
                collider.invulberable_frame = cfg.TANK_LONG_INVULNERABLE_FRAMES
                print("same", collider.same_collider_counter)

    def regen_health(self):
        if not self.alive:
            return
        if self.hp_regen_frame > 0:
            self.hp += self.max_hp * self.slow_health_regen
        elif self.hp < self.max_hp:
            self.hp += self.max_hp * self.fast_health_regen
        if self.hp > self.max_hp:
                self.hp = self.max_hp

    def update_counter(self):
        if self.invulberable_frame > 0:
            self.invulberable_frame -= 1
        if self.hp_regen_frame > 0:
            self.hp_regen_frame -= 1

    def move(self, dx: float, dy: float):
        if not self.alive:
            return

        # Handle normal motion
        max_v = cfg.BASE_MAX_VELOCITY * self.v_scale

        if dx != 0 or dy != 0:
            magnitude = np.hypot(dx, dy)
            dx, dy = dx / magnitude, dy / magnitude
            self.ax = dx * max_v / cfg.BASE_ACC_FRAMES
            self.ay = dy * max_v / cfg.BASE_ACC_FRAMES
        else:
            speed = np.hypot(self.vx, self.vy)
            if speed < 1e-6:
                self.vx = 0.0
                self.vy = 0.0
                self.ax = 0.0
                self.ay = 0.0
            elif speed > 0:
                dx, dy = -self.vx / speed, -self.vy / speed
                self.ax = dx * max_v / cfg.BASE_DEC_FRAMES
                self.ay = dy * max_v / cfg.BASE_DEC_FRAMES

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
        else:
            self.collision_frame = 0
            self.collision_vx = 0.0
            self.collision_vy = 0.0

        # Recoil
        if self.type == UnitType.Tank:
            recoil_vx = self.recoil_vx * cfg.TANK_RECOIL_V_SCALE
            recoil_vy = self.recoil_vy * cfg.TANK_RECOIL_V_SCALE

            # decay speed
            self.recoil_vx *= cfg.TANK_RECOIL_DECAY
            self.recoil_vy *= cfg.TANK_RECOIL_DECAY
        else:
            recoil_vx = recoil_vy = 0.0

        final_vx = self.vx + collision_vx + recoil_vx
        final_vy = self.vy + collision_vy + recoil_vy

        self.x += final_vx
        self.y += final_vy

        self.x = np.clip(self.x, self.radius, cfg.MAP_SIZE - self.radius)
        self.y = np.clip(self.y, self.radius, cfg.MAP_SIZE - self.radius)
