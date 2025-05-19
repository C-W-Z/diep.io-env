import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys
from enum import Enum

class UnitType(Enum):
    Tank = 0
    Polygon = 1
    Bullet = 2

# Constants
MAP_SIZE = 1000.0
BASE_MAX_VELOCITY = 5.0
BASE_MAX_ACCELLERATION = 0.5
COLLISION_BOUNCE_V_SCALE = 0.5
COLLISION_BOUNCE_DECCELLERATE_FRAMES = 30

BASE_ACCELLERATION_FRAMES = 20
BASE_DECCELLERATION_FRAMES = 20

INVULNERABLE_FRAMES = 5

OBSERVATION_SIZE = 1000

SLOW_HP_REGEN_FRAMES = 30 * 60 # 30 seconds in 60 fps

class Unit:
    def __init__(
        self,
        unit_type=UnitType.Polygon,
        x=np.random.uniform(0, MAP_SIZE),
        y=np.random.uniform(0, MAP_SIZE),
        max_hp=50.0,
    ):
        self.x = x
        self.y = y
        self.max_hp = max_hp
        self.hp = max_hp
        self.v_scale = 1.0
        self.radius = 20.0
        self.angle = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.collision_frame = 0
        self.collision_vx = 0.0
        self.collision_vy = 0.0
        self.invulberable_frame = 0
        self.hp_regen_frames = 0

        self.type = unit_type

        # stats
        self.health_regen = 0
        self.body_damage = 20.0

    @property
    def alive(self):
        return self.hp > 0

    def recv_damage(self, damage, collider_hp, collider_type):
        body_dmg = self.body_damage * (0.25 if collider_type == UnitType.Bullet else 1.0)
        damage_scale = 1.0
        if self.type == UnitType.Bullet:
            damage_scale = 0.25
        elif self.type == UnitType.Tank and collider_type == UnitType.Tank:
            damage_scale = 1.5
        if collider_hp >= body_dmg:
            self.hp -= damage_scale * damage
        else:
            self.hp -= damage_scale * damage * collider_hp / body_dmg
        if self.hp < 0:
            self.hp = 0.0
        self.hp_regen_frames = SLOW_HP_REGEN_FRAMES
        self.invulberable_frame = INVULNERABLE_FRAMES

    def update_counter(self):
        if self.invulberable_frame > 0:
            self.invulberable_frame -= 1
        if self.hp_regen_frames > 0:
            self.hp_regen_frames -= 1

    def update(self, dx, dy):
        if not self.alive:
            return

        # Handle normal motion
        if dx != 0 or dy != 0:
            magnitude = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / magnitude, dy / magnitude
            self.ax = dx * BASE_MAX_VELOCITY / BASE_ACCELLERATION_FRAMES
            self.ay = dy * BASE_MAX_VELOCITY / BASE_ACCELLERATION_FRAMES
        else:
            speed = np.sqrt(self.vx**2 + self.vy**2)
            if speed < 0.1:
                self.vx = 0.0
                self.vy = 0.0
                self.ax = 0.0
                self.ay = 0.0
            elif speed > 0:
                dx, dy = -self.vx / speed, -self.vy / speed
                self.ax = dx * BASE_MAX_VELOCITY / BASE_DECCELLERATION_FRAMES
                self.ay = dy * BASE_MAX_VELOCITY / BASE_DECCELLERATION_FRAMES

        self.vx += self.ax
        self.vy += self.ay
        normal_speed = np.sqrt(self.vx**2 + self.vy**2)
        max_v = BASE_MAX_VELOCITY * self.v_scale
        if normal_speed > max_v:
            self.vx = self.vx * max_v / normal_speed
            self.vy = self.vy * max_v / normal_speed

        # Handle collision motion
        collision_vx, collision_vy = 0.0, 0.0
        if self.collision_frame > 0:
            factor = self.collision_frame / COLLISION_BOUNCE_DECCELLERATE_FRAMES
            collision_vx = self.collision_vx * factor
            collision_vy = self.collision_vy * factor
            self.collision_frame -= 1
        else:
            self.collision_frame = 0
            self.collision_vx = 0.0
            self.collision_vy = 0.0

        final_vx = self.vx + collision_vx
        final_vy = self.vy + collision_vy
        # final_speed = np.sqrt(final_vx**2 + final_vy**2)
        # if final_speed > max_v and final_speed > 0:
        #     final_vx = final_vx * max_v / final_speed
        #     final_vy = final_vy * max_v / final_speed

        self.x += final_vx
        self.y += final_vy
        self.x = np.clip(self.x, self.radius, MAP_SIZE - self.radius)
        self.y = np.clip(self.y, self.radius, MAP_SIZE - self.radius)

class Player(Unit):
    def __init__(
            self,
            x=np.random.uniform(0, MAP_SIZE),
            y=np.random.uniform(0, MAP_SIZE),
            max_hp=50.0,
            score=0,
        ):
        super(Player, self).__init__(x=x, y=y, max_hp=max_hp)
        self.score = score

class DiepIOEnvBasic(gym.Env):
    def __init__(self, num_players=2, render_mode=True):
        super(DiepIOEnvBasic, self).__init__()
        self.num_players = num_players
        self.render_mode = render_mode
        self.max_steps = 1000000

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_players * 6,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32
        )

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((OBSERVATION_SIZE, OBSERVATION_SIZE))
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.units = [
            Unit(x=np.random.uniform(100, MAP_SIZE-100), y=np.random.uniform(100, MAP_SIZE-100), max_hp=50.0)
            for _ in range(self.num_players)
        ]
        obs = {i: self._get_obs(i) for i in range(self.num_players) if self.units[i].alive}
        if self.render_mode:
            self.render()
        return obs, {}

    def _get_obs(self, agent_id):
        # Create a 500x500 surface for observation
        obs_surface = pygame.Surface((OBSERVATION_SIZE, OBSERVATION_SIZE))
        obs_surface.fill((255, 255, 255))  # White background

        # Calculate scale: 500 pixels cover 833.33 map units
        obs_scale = 1
        obs_half = OBSERVATION_SIZE // 2  # 250

        # Center on the agent
        agent = self.units[agent_id]
        center_x, center_y = agent.x, agent.y

        # Draw black rectangles for areas outside map boundaries (0, 0, 1000, 1000)
        # Calculate map boundaries in pixel coordinates
        left_boundary = int(obs_half + (0 - center_x) * obs_scale)  # x = 0
        right_boundary = int(obs_half + (MAP_SIZE - center_x) * obs_scale)  # x = 1000
        top_boundary = int(obs_half + (0 - center_y) * obs_scale)  # y = 0
        bottom_boundary = int(obs_half + (MAP_SIZE - center_y) * obs_scale)  # y = 1000

        # Draw black rectangles for out-of-bounds areas
        black_color = (0, 0, 0)
        # Left (x < 0)
        if left_boundary > 0:
            pygame.draw.rect(obs_surface, black_color, (0, 0, left_boundary, OBSERVATION_SIZE))
        # Right (x > 1000)
        if right_boundary < OBSERVATION_SIZE:
            pygame.draw.rect(obs_surface, black_color, (right_boundary, 0, OBSERVATION_SIZE - right_boundary, OBSERVATION_SIZE))
        # Top (y < 0)
        if top_boundary > 0:
            pygame.draw.rect(obs_surface, black_color, (0, 0, OBSERVATION_SIZE, top_boundary))
        # Bottom (y > 1000)
        if bottom_boundary < OBSERVATION_SIZE:
            pygame.draw.rect(obs_surface, black_color, (0, bottom_boundary, OBSERVATION_SIZE, OBSERVATION_SIZE - bottom_boundary))

        # Draw all units (same as original render)
        for i, unit in enumerate(self.units):
            if not unit.alive:
                continue
            # Calculate relative position in map units
            rel_x = unit.x - center_x
            rel_y = unit.y - center_y
            # Convert to pixel coordinates in 500x500 observation
            pixel_x = int(obs_half + rel_x * obs_scale)
            pixel_y = int(obs_half + rel_y * obs_scale)
            # Draw unit
            if (0 <= pixel_x < OBSERVATION_SIZE and 0 <= pixel_y < OBSERVATION_SIZE):
                color = (255, 0, 0) if i == 0 else (0, 0, 255)
                pygame.draw.circle(
                    obs_surface, color,
                    (pixel_x, pixel_y),
                    int(unit.radius * obs_scale)
                )
                # Draw orientation line
                end_x = pixel_x + unit.radius * 2 * obs_scale * np.cos(unit.angle)
                end_y = pixel_y - unit.radius * 2 * obs_scale * np.sin(unit.angle)  # Flip sin for Pygame y-axis
                pygame.draw.line(
                    obs_surface, (0, 0, 0),
                    (pixel_x, pixel_y),
                    (end_x, end_y),
                    2
                )
                # Draw HP bar
                hp_width = int(20 * unit.hp / unit.max_hp)
                pygame.draw.rect(
                    obs_surface, (0, 255, 0),
                    (pixel_x - 10, pixel_y - 15, hp_width, 3)
                )

        # Convert surface to RGB NumPy array
        obs_array = pygame.surfarray.array3d(obs_surface)  # Shape: (500, 500, 3)
        return obs_array

    def _get_player_input(self):
        dx, dy, shoot = 0.0, 0.0, 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            dy -= 1.0
        if keys[pygame.K_s]:
            dy += 1.0
        if keys[pygame.K_a]:
            dx -= 1.0
        if keys[pygame.K_d]:
            dx += 1.0
        if dx != 0 or dy != 0:
            magnitude = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / magnitude, dy / magnitude
        mouse_x, mouse_y = pygame.mouse.get_pos()
        scale = 1
        mouse_x, mouse_y = mouse_x / scale, mouse_y / scale
        self.units[0].angle = np.arctan2(OBSERVATION_SIZE // 2 - mouse_y, mouse_x - OBSERVATION_SIZE // 2)
        # print(self.units[0].angle)
        return np.array([dx, dy, shoot], dtype=np.float32)

    def _get_random_input(self):
        dx, dy, shoot = 0.0, 0.0, 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            dy -= 1.0
        if keys[pygame.K_DOWN]:
            dy += 1.0
        if keys[pygame.K_LEFT]:
            dx -= 1.0
        if keys[pygame.K_RIGHT]:
            dx += 1.0
        if dx != 0 or dy != 0:
            magnitude = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / magnitude, dy / magnitude
        self.units[1].angle = np.random.uniform(0, 2 * np.pi)
        return np.array([dx, dy, shoot], dtype=np.float32)

    def _handle_collisions(self):
        for i in range(self.num_players):
            for j in range(i + 1, self.num_players):
                if not (self.units[i].alive and self.units[j].alive):
                    continue
                dx = self.units[i].x - self.units[j].x
                dy = self.units[i].y - self.units[j].y
                distance = np.sqrt(dx**2 + dy**2)
                if distance < (self.units[i].radius + self.units[j].radius):
                    if distance == 0:
                        dx, dy = 1.0, 0.0
                        distance = 1.0
                    nx, ny = dx / distance, dy / distance
                    max_v = BASE_MAX_VELOCITY * COLLISION_BOUNCE_V_SCALE
                    # overlap = 2 * self.units[i].radius - distance
                    # self.units[i].x += nx * overlap / 2
                    # self.units[i].y += ny * overlap / 2
                    # self.units[j].x -= nx * overlap / 2
                    # self.units[j].y -= ny * overlap / 2
                    if self.units[i].invulberable_frame == 0:
                        self.units[i].collision_vx = nx * max_v
                        self.units[i].collision_vy = ny * max_v
                        self.units[i].collision_frame = COLLISION_BOUNCE_DECCELLERATE_FRAMES
                        self.units[i].recv_damage(self.units[j].body_damage, self.units[j].hp, self.units[j].type)
                    if self.units[j].invulberable_frame == 0:
                        self.units[j].collision_vx = -nx * max_v
                        self.units[j].collision_vy = -ny * max_v
                        self.units[j].collision_frame = COLLISION_BOUNCE_DECCELLERATE_FRAMES
                        self.units[j].recv_damage(self.units[i].body_damage, self.units[i].hp, self.units[i].type)
                    print("collision")

    def step(self, actions=None):
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.num_players)}
        dones = {i: False for i in range(self.num_players)}
        infos = {}

        if actions is None:
            actions = {}
            if self.units[0].alive:
                actions[0] = self._get_player_input()
            if self.num_players > 1 and self.units[1].alive:
                actions[1] = self._get_random_input()

        for i, action in actions.items():
            if self.units[i].alive:
                self.units[i].update_counter()
                dx, dy, _ = action
                self.units[i].update(dx, dy)
                rewards[i] += 0.01

        self._handle_collisions()

        for i in range(self.num_players):
            if not self.units[i].alive:
                dones[i] = True
                rewards[i] -= 10.0
        if self.step_count >= self.max_steps or sum(unit.alive for unit in self.units) <= 1:
            dones = {i: True for i in range(self.num_players)}

        obs = {i: self._get_obs(i) for i in range(self.num_players) if self.units[i].alive}
        if self.render_mode:
            self.render()
        return obs, rewards, dones, False, infos

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill((255, 255, 255))  # White background

        # Display observation for each alive agent
        for i, unit in enumerate(self.units[:1]):
            if not unit.alive:
                continue
            # Get observation (500x500 RGB)
            obs = self._get_obs(i)  # Shape: (500, 500, 3)
            # Create Pygame surface
            obs_surface = pygame.surfarray.make_surface(obs)
            # Draw at appropriate position
            x_pos = i * OBSERVATION_SIZE  # Agent 0 at (0, 0), Agent 1 at (500, 0)
            self.screen.blit(obs_surface, (x_pos, 0))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    env = DiepIOEnvBasic(num_players=2, render_mode=True)
    obs, _ = env.reset()
    while True:
        obs, rewards, dones, _, _ = env.step()
        if all(dones.values()):
            break
    env.close()