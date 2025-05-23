import numpy as np
from enum import IntEnum
from config import config as cfg
from unit import Unit, UnitType

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
        i     = int(key)
        shift = i * 4
        mask  = 0b1111 << shift

        return (self.raw & mask) >> shift

    def __setitem__(self, key, value):
        assert value >= 0 and value <= 7

        i     = int(key)
        shift = i * 4
        mask  = 0b1111 << shift

        self.raw = (self.raw & ~mask) | (value << shift)

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
        max_hp=50.0,
        score=0,
    ):
        super(Tank, self).__init__(
            unit_type=UnitType.Tank,
            x=x,
            y=y,
            max_hp=max_hp,
            score=score,
        )
        self.level = self.score2level(self.score)
        self.skill_points = self.level2sp(self.level)
        self.stats = TankStats()
        self.calc_stats_properties()
        self.reload_counter = 0

    def add_score(self, score):
        self.score += score
        old_level = self.level
        self.level = self.score2level(self.score)
        self.skill_points += self.level2sp(self.level) - self.level2sp(old_level)
        self.calc_stats_properties()

    def add_points(self, i: int):
        if self.skill_points == 0:
            return False
        if self.stats.add_point(i):
            self.skill_points -= 1
            self.calc_stats_properties()
            return True
        return False

    @staticmethod
    def score2level(score: int):
        level = 1
        for i, exp in enumerate(cfg.EXP_LIST):
            if score < exp:
                break
            level = i
        return level

    @staticmethod
    def level2sp(level: int): # level to skill points
        if level <= 28:
            return level - 1
        else:
            return 27 + (level - 27) // 3

    def calc_stats_properties(self):
        # Health Regen
        self.slow_health_regen = (0.03 + 0.12 * self.stats[TST.HealthRegen]) / 30 / cfg.FPS
        self.fast_health_regen = cfg.FAST_REGEN_LIST[self.stats[TST.HealthRegen]] / cfg.FPS

        # Max Health
        old_max_hp = self.max_hp
        self.max_hp = 50.0 + 2 * (self.level - 1) + self.stats[TST.MaxHealth] * 20.0
        if self.max_hp > old_max_hp:
            self.hp += self.max_hp - old_max_hp
        elif self.hp > self.max_hp:
            self.hp = self.max_hp

        # Body Damage
        self.body_damage = 20.0 + self.stats[TST.BodyDamage] * 4.0

        # Bullet Speed
        self.bullet_v_scale = cfg.BASE_BULLET_V_SCALE # TODO

        # Bullet Penetration
        # self.bullet_max_hp = 2.0 + self.stats[TST.BulletPen] * 1.5
        self.bullet_max_hp = 2.0 + 7 * 1.5

        # Bullet Damage
        self.bullet_damage = 7.0 + self.stats[TST.BulletDamage] * 3.0

        # Reload
        self.reload_frames = (0.6 - self.stats[TST.Reload] * 0.04) * cfg.FPS

        # Movement Speed
        self.v_scale = 1.0 + self.stats[TST.Speed] * 0.03 - (self.level - 1) * 0.001 # not sure

        # ===== Hidden Properties =====

        # tank size
        self.radius = 1.0 * np.pow(1.01, (self.level - 1))

        # bullet size
        self.bullet_radius = 0.5 * np.pow(1.01, (self.level - 1))

        # Recoil

        # FOV
        self.observation_size = 40.0 + (self.level - 1) * 10.0 / 44

        # Knockback Resistance
        # self.stats[TST.BodyDamage], self.stats[TST.Speed]

    def update_counter(self):
        """Update invulnerability, HP-regen, and reload counters each frame."""
        if self.invulberable_frame > 0:
            self.invulberable_frame -= 1
        if self.hp_regen_frame > 0:
            self.hp_regen_frame -= 1
        if self.reload_counter > 0:
            self.reload_counter -= 1
