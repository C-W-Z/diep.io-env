import numpy as np
from enum import IntEnum
from .config import config as cfg
from .unit import Unit, UnitType
from numba import njit

class TST(IntEnum):
    HealthRegen  = 0
    MaxHealth    = 1
    BodyDamage   = 2
    BulletSpeed  = 3
    BulletPen    = 4
    BulletDamage = 5
    Reload       = 6
    Speed        = 7

class TankStats:
    def __init__(self):
        self.raw = np.uint32(0)

    def __getitem__(self, key):
        return get_stat(self.raw, int(key))

    def __setitem__(self, key, value):
        assert 0 <= value <= 7

        value = np.uint32(value)
        i     = np.uint32(key)
        shift = i * 4
        # build a 32-bit mask for the 4 bits we want to clear/set
        mask     = np.uint32(0b1111) << shift
        inv_mask = ~mask               # bitwise NOT within uint32

        # clear the 4 bits at position, then OR in the new value
        self.raw = (self.raw & inv_mask) | (value << shift)

    def add_point(self, attribute_type):
        if self[attribute_type] >= 7:
            return False

        self[attribute_type] += 1
        return True

class Tank(Unit):
    def __init__(
        self,
        x,
        y,
        score=0,
    ):
        super(Tank, self).__init__(
            unit_type=UnitType.Tank,
            x=x,
            y=y,
            max_hp=50.0,
            score=score,
        )
        self.level = score2level(self.score, cfg.EXP_LIST)
        self.skill_points = level2sp(self.level)
        self.stats = TankStats()
        self.calc_stats_properties()
        self.reload_counter = 0

        self.same_collider_counter_reset_frame = 0
        self.last_collider_id = -1
        self.same_collider_counter = 0

    def update_counter(self):
        super(Tank, self).update_counter()
        if self.reload_counter > 0:
            self.reload_counter -= 1
        self.same_collider_counter_reset_frame -= 1
        if self.same_collider_counter_reset_frame <= 0:
            self.same_collider_counter = 0

    def add_score(self, score: int):
        self.score += score
        old_level = self.level
        self.level = score2level(self.score, cfg.EXP_LIST)
        self.skill_points += level2sp(self.level) - level2sp(old_level)
        self.calc_stats_properties()

    def reset_score(self, score):
        old_level = self.level
        self.score = score
        self.level = score2level(self.score, cfg.EXP_LIST)
        if self.level < old_level:
            self.calc_stats_properties()

    def add_points(self, i: int):
        if self.skill_points == 0:
            return False
        if self.stats.add_point(i):
            self.skill_points -= 1
            self.calc_stats_properties()
            return True
        return False

    def calc_stats_properties(self):

        (
            self.slow_health_regen, self.fast_health_regen, self.max_hp, self.hp, self.body_damage,
            self.bullet_v_scale, self.bullet_max_hp, self.bullet_damage, self.reload_frames, self.v_scale,
            self.radius, self.bullet_radius, self.observation_size
        ) = calc_tank_stats_properties(
            self.stats.raw, self.level, self.max_hp, self.hp, cfg.FAST_REGEN_LIST, cfg.FPS, cfg.BASE_BULLET_V_SCALE
        )

    def calc_respawn_score(self):
        if self.alive:
            return
        self.score = cfg.EXP_LIST[cfg.RESPAWN_LEVEL_LIST[self.level]]

@njit
def get_stat(stats, index):
    """Extract a 4-bit stat from a uint32 using bitwise operations."""
    shift = index * 4
    mask = 0b1111 << shift
    return (stats & mask) >> shift

@njit
def calc_tank_stats_properties(stats, level, max_hp, hp, fast_regen_list, fps, base_bullet_v_scale):
    # Health Regen
    slow_health_regen = (0.03 + 0.12 * get_stat(stats, 0)) / 30 / fps
    fast_health_regen = fast_regen_list[get_stat(stats, 0)] / fps
    # Max Health
    new_max_hp = 10000.0 + 2 * (level - 1) + get_stat(stats, 1) * 20.0
    hp = hp + (new_max_hp - max_hp) if new_max_hp > max_hp else min(hp, new_max_hp)
    # Body Damage
    body_damage = 20.0 + get_stat(stats, 2) * 4.0
    # Bullet Speed
    bullet_v_scale = base_bullet_v_scale + get_stat(stats, 3) * 0.06  # TODO
    # Bullet Penetration
    bullet_max_hp = 2.0 + get_stat(stats, 4) * 1.5
    # Bullet Damage
    bullet_damage = 7.0 + get_stat(stats, 5) * 3.0
    # Reload
    reload_frames = (0.6 - get_stat(stats, 6) * 0.04) * fps
    # Movement Speed
    v_scale = 1.0 + get_stat(stats, 7) * 0.03 - (level - 1) * 0.001

    # ===== Hidden Properties =====

    # tank size
    radius = 1.0 * np.pow(1.01, level - 1)
    # bullet size
    bullet_radius = 0.5 * np.pow(1.01, level - 1)
    # Recoil
    # TODO
    # FOV
    observation_size = 40.0 + (level - 1) * 10.0 / 44
    # Knockback Resistance  # TODO
    # self.stats[TST.BodyDamage], self.stats[TST.Speed]

    return (slow_health_regen, fast_health_regen, new_max_hp, hp, body_damage,
            bullet_v_scale, bullet_max_hp, bullet_damage, reload_frames, v_scale,
            radius, bullet_radius, observation_size)

@njit
def score2level(score, exp_list):
    level = 1
    for i, exp in enumerate(exp_list):
        if score < exp:
            break
        level = i
    return level

@njit
def level2sp(level):
    if level <= 28:
        return level - 1
    return 27 + (level - 27) // 3