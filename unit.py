import numpy as np
from enum import IntEnum
from config import config as cfg
import itertools
from numba import njit
from utils import clip_scalar

class UnitType(IntEnum):
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
        self.total_vx, self.total_vy = 0.0, 0.0
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

    def add_score(self, score: int):
        # Bullet, Tank should overwrite
        self.score += score

    def deal_damage(self, collider: "Unit", self_hp_before_hit = None):
        if self_hp_before_hit == None:
            self_hp_before_hit = self.hp

        collider.hp, score_to_add, collider.hp_regen_frame, collider.invulberable_frame = deal_damage_unit(
            self_hp_before_hit, self.body_damage, self.type,
            collider.hp, collider.body_damage, collider.type, collider.score,
            cfg.EXP_LIST[-1], cfg.SLOW_HP_REGEN_FRAMES, cfg.INVULNERABLE_FRAMES, cfg.TANK_LONG_INVULNERABLE_FRAMES
        )

        if score_to_add != 0:
            self.add_score(score_to_add)

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

    # 在 Unit 类的 move 方法中调用
    def move(self, dx: float, dy: float):
        if not self.alive:
            return
        max_v = cfg.BASE_MAX_VELOCITY * self.v_scale
        self.x, self.y, self.vx, self.vy, self.total_vx, self.total_vy, self.ax, self.ay, self.collision_vx, self.collision_vy, \
        self.collision_frame, self.recoil_vx, self.recoil_vy = move_unit(
            self.x, self.y, dx, dy, self.vx, self.vy, self.ax, self.ay, self.radius,
            self.collision_vx, self.collision_vy, self.collision_frame, self.max_collision_frame,
            self.recoil_vx, self.recoil_vy, max_v, cfg.BASE_ACC_FRAMES, cfg.BASE_DEC_FRAMES,
            cfg.TANK_RECOIL_V_SCALE if self.type == UnitType.Tank else 0.0,
            cfg.TANK_RECOIL_DECAY if self.type == UnitType.Tank else 0.0, cfg.MAP_SIZE
        )

@njit
def move_unit(x, y, dx, dy, vx, vy, ax, ay, radius, collision_vx, collision_vy,
              collision_frame, max_collision_frame, recoil_vx, recoil_vy, max_v, base_acc_frames,
              base_dec_frames, tank_recoil_v_scale, tank_recoil_decay, map_size):
    # Handle normal motion
    if dx != 0 or dy != 0:
        magnitude = np.hypot(dx, dy)
        dx, dy = dx / magnitude, dy / magnitude
        ax = dx * max_v / base_acc_frames
        ay = dy * max_v / base_acc_frames
    else:
        speed = np.hypot(vx, vy)
        if speed < 1e-6:
            vx, vy, ax, ay = 0.0, 0.0, 0.0, 0.0
        elif speed > 0:
            dx, dy = -vx / speed, -vy / speed
            ax = dx * max_v / base_dec_frames
            ay = dy * max_v / base_dec_frames

    vx += ax
    vy += ay
    normal_speed = np.hypot(vx, vy)
    if normal_speed > max_v:
        vx = vx * max_v / normal_speed
        vy = vy * max_v / normal_speed

    # Handle collision motion
    if collision_frame > 0:
        factor = collision_frame / max_collision_frame
        collision_vx = collision_vx * factor
        collision_vy = collision_vy * factor
        collision_frame -= 1
    else:
        collision_frame, collision_vx, collision_vy = 0, 0.0, 0.0

    # Recoil (simplified for Tank)
    _recoil_vx = recoil_vx * tank_recoil_v_scale
    _recoil_vy = recoil_vy * tank_recoil_v_scale
    recoil_vx *= tank_recoil_decay
    recoil_vy *= tank_recoil_decay

    total_vx = vx + collision_vx + _recoil_vx
    total_vy = vy + collision_vy + _recoil_vy

    x += total_vx
    y += total_vy

    x = clip_scalar(x, radius, map_size - radius)
    y = clip_scalar(y, radius, map_size - radius)

    return x, y, vx, vy, total_vx, total_vy, ax, ay, collision_vx, collision_vy, collision_frame, recoil_vx, recoil_vy

@njit
def deal_damage_unit(self_hp, self_body_damage, self_type,
                     collider_hp, collider_body_damage, collider_type, collider_score,
                     max_exp, slow_hp_regen_frames, invulnerable_frames, tank_long_invulnerable_frames):
    self_hp_before_hit = self_hp
    body_dmg = collider_body_damage * (0.25 if self_type == 2 else 1.0)  # UnitType.Bullet = 2
    damage_scale = 1.0
    if collider_type == 2:  # Bullet
        damage_scale = 0.25
    elif collider_type == 0 and self_type == 0:  # Tank vs Tank
        damage_scale = 1.5

    if self_hp_before_hit >= body_dmg:
        collider_hp -= damage_scale * self_body_damage
    else:
        collider_hp -= damage_scale * self_body_damage * self_hp_before_hit / body_dmg

    score_to_add = 0
    if collider_hp < 0:
        collider_hp = 0.0
        score_to_add = min(collider_score, max_exp)
    else:
        hp_regen_frame = slow_hp_regen_frames
        invulnerable_frame = invulnerable_frames
        if collider_type == 0 and self_type == 0:  # Tank vs Tank
            invulnerable_frame = tank_long_invulnerable_frames
    return collider_hp, score_to_add, hp_regen_frame, invulnerable_frame
