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
MAP_SIZE = 80
BOARDER_SIZE = 5
BASE_MAX_VELOCITY = 0.5
BASE_MAX_ACCELLERATION = 0.02
COLLISION_BOUNCE_V_SCALE = 0.75
COLLISION_BOUNCE_DECCELLERATE_FRAMES = 40
BASE_ACCELLERATION_FRAMES = 20
BASE_DECCELLERATION_FRAMES = 30
INVULNERABLE_FRAMES = 5
SCREEN_SIZE = 1000  # Pixel size of render window
SLOW_HP_REGEN_FRAMES = 30 * 60  # 30 seconds in 60 fps

class Unit:
    def __init__(
        self,
        unit_type=UnitType.Polygon,
        x=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
        y=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
        max_hp=50.0,
    ):
        self.x = x
        self.y = y
        self.max_hp = max_hp
        self.hp = max_hp
        self.v_scale = 1.0
        self.radius = 1
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
        # Stats
        self.health_regen = 0
        self.body_damage = 20.0

    @property
    def alive(self):
        return self.hp > 0

    def recv_damage(self, damage: float, collider_hp: float, collider_type: UnitType):
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

        self.x += final_vx
        self.y += final_vy
        self.x = np.clip(self.x, self.radius, MAP_SIZE - self.radius)
        self.y = np.clip(self.y, self.radius, MAP_SIZE - self.radius)

class Tank(Unit):
    def __init__(
        self,
        x=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
        y=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
        max_hp=50.0,
        score=0,
    ):
        super(Tank, self).__init__(
            unit_type=UnitType.Tank,
            x=x,
            y=y,
            max_hp=max_hp
        )
        self.score = score
        self.observation_size = 40.0

class DiepIOEnvBasic(gym.Env):
    def __init__(self, n_tanks=2, render_mode=True):
        super(DiepIOEnvBasic, self).__init__()
        self.n_tanks = n_tanks
        self.render_mode = render_mode
        self.max_steps = 1000000

        # Observation space: Vector of tank states
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_tanks * 6,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32
        )

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            self.clock = pygame.time.Clock()
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(SCREEN_SIZE, SCREEN_SIZE, 3), dtype=np.uint8
            )

        self.reset()

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.tanks = [
            Tank(
                x=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
                y=np.random.uniform(BOARDER_SIZE, MAP_SIZE-BOARDER_SIZE),
                max_hp=50.0,
                score=0,
            )
            for _ in range(self.n_tanks)
        ]
        obs = {i: self._get_obs(i) for i in range(self.n_tanks) if self.tanks[i].alive}
        if self.render_mode:
            self.render()
        return obs, {}

    def _get_obs(self, agent_id):
        # Placeholder: Current vector observation (not used for rendering)
        obs = []
        for i in range(self.n_tanks):
            tank = self.tanks[i]
            if tank.alive:
                obs.extend([tank.x, tank.y, tank.vx, tank.vy, tank.hp, tank.angle])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def _render_frame(self, agent_id):
        # Create a SCREEN_SIZE x SCREEN_SIZE surface
        surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        surface.fill((255, 255, 255))  # White background

        # Center on the agent
        agent = self.tanks[agent_id]
        center_x, center_y = agent.x, agent.y
        observation_size = agent.observation_size

        # Calculate scale: SCREEN_SIZE pixels cover observation_size map units
        grid_size = SCREEN_SIZE / observation_size  # e.g., 1000 / 40 = 25
        screen_half = SCREEN_SIZE // 2  # 500

        # Draw black rectangles for areas outside map boundaries
        left_boundary = int(screen_half + (BOARDER_SIZE - center_x) * grid_size)  # x = BOARDER_SIZE
        right_boundary = int(screen_half + (MAP_SIZE - BOARDER_SIZE - center_x) * grid_size)  # x = MAP_SIZE - BOARDER_SIZE
        top_boundary = int(screen_half + (BOARDER_SIZE - center_y) * grid_size)  # y = BOARDER_SIZE
        bottom_boundary = int(screen_half + (MAP_SIZE - BOARDER_SIZE - center_y) * grid_size)  # y = MAP_SIZE - BOARDER_SIZE

        black_color = (191, 191, 191)
        if left_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, left_boundary, SCREEN_SIZE))
        if right_boundary < SCREEN_SIZE:
            pygame.draw.rect(surface, black_color, (right_boundary, 0, SCREEN_SIZE - right_boundary, SCREEN_SIZE))
        if top_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, SCREEN_SIZE, top_boundary))
        if bottom_boundary < SCREEN_SIZE:
            pygame.draw.rect(surface, black_color, (0, bottom_boundary, SCREEN_SIZE, SCREEN_SIZE - bottom_boundary))

        # Draw grid lines (spacing = 1 map units)
        grid_color = (150, 150, 150)  # Light gray
        # Calculate visible grid lines
        min_x = max(0, center_x - observation_size / 2)
        max_x = min(MAP_SIZE, center_x + observation_size / 2)
        min_y = max(0, center_y - observation_size / 2)
        max_y = min(MAP_SIZE, center_y + observation_size / 2)
        # Grid line positions
        x_grid = np.arange(np.ceil(min_x), np.floor(max_x) + 1)
        y_grid = np.arange(np.ceil(min_y), np.floor(max_y) + 1)

        left_boundary = int(screen_half + (0 - center_x) * grid_size)  # x = 0
        right_boundary = int(screen_half + (MAP_SIZE - center_x) * grid_size)  # x = MAP_SIZE
        top_boundary = int(screen_half + (0 - center_y) * grid_size)  # y = 0
        bottom_boundary = int(screen_half + (MAP_SIZE - center_y) * grid_size)  # y = MAP_SIZE

        # Draw vertical lines
        for x in x_grid:
            pixel_x = int(screen_half + (x - center_x) * grid_size)
            if 0 <= pixel_x < SCREEN_SIZE:
                pygame.draw.line(surface, grid_color, (pixel_x, top_boundary), (pixel_x, bottom_boundary), 1)
        # Draw horizontal lines
        for y in y_grid:
            pixel_y = int(screen_half + (y - center_y) * grid_size)
            if 0 <= pixel_y < SCREEN_SIZE:
                pygame.draw.line(surface, grid_color, (left_boundary, pixel_y), (right_boundary, pixel_y), 1)

        # Draw all tanks
        for i, unit in enumerate(self.tanks):
            if not unit.alive:
                continue
            rel_x = unit.x - center_x
            rel_y = unit.y - center_y
            pixel_x = int(screen_half + rel_x * grid_size)
            pixel_y = int(screen_half + rel_y * grid_size)
            if 0 <= pixel_x < SCREEN_SIZE and 0 <= pixel_y < SCREEN_SIZE:
                # Draw HP bar
                hp_width = int(grid_size * 2 * unit.radius * unit.hp / unit.max_hp)  # Scale with grid
                pygame.draw.rect(
                    surface, (0, 216, 0),
                    (pixel_x - grid_size * 1, pixel_y + grid_size * 1.2, hp_width, 5)
                )
                # Draw orientation line
                end_x = pixel_x + unit.radius * 2 * grid_size * np.cos(unit.angle)
                end_y = pixel_y - unit.radius * 2 * grid_size * np.sin(unit.angle)  # Flip for Pygame y-axis
                pygame.draw.line(
                    surface, (0, 0, 0),
                    (pixel_x, pixel_y),
                    (end_x, end_y),
                    2
                )
                # Draw body
                color = (0, 127, 255) if i == 0 else (255, 0, 0)
                pygame.draw.circle(
                    surface, color,
                    (pixel_x, pixel_y),
                    int(unit.radius * grid_size)
                )

        return surface
        # Convert surface to RGB NumPy array
        # obs_array = pygame.surfarray.array3d(obs_surface)  # Shape: (500, 500, 3)
        # return obs_array

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
        screen_half = SCREEN_SIZE // 2
        self.tanks[0].angle = np.arctan2(screen_half - mouse_y, mouse_x - screen_half)
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
        self.tanks[1].angle = np.random.uniform(0, 2 * np.pi)
        return np.array([dx, dy, shoot], dtype=np.float32)

    def _handle_collisions(self):
        for i in range(self.n_tanks):
            for j in range(i + 1, self.n_tanks):
                if not (self.tanks[i].alive and self.tanks[j].alive):
                    continue
                dx = self.tanks[i].x - self.tanks[j].x
                dy = self.tanks[i].y - self.tanks[j].y
                distance = np.sqrt(dx**2 + dy**2)
                radius_sum = self.tanks[i].radius + self.tanks[j].radius
                if distance < radius_sum:
                    if distance == 0:
                        dx, dy = 1.0, 0.0
                        distance = 1.0
                    nx, ny = dx / distance, dy / distance
                    max_v = BASE_MAX_VELOCITY * COLLISION_BOUNCE_V_SCALE
                    unit_i_hp_before_hit = self.tanks[i].hp
                    if self.tanks[i].invulberable_frame == 0:
                        self.tanks[i].collision_vx = nx * max_v
                        self.tanks[i].collision_vy = ny * max_v
                        self.tanks[i].collision_frame = COLLISION_BOUNCE_DECCELLERATE_FRAMES
                        self.tanks[i].recv_damage(self.tanks[j].body_damage, self.tanks[j].hp, self.tanks[j].type)
                    if self.tanks[j].invulberable_frame == 0:
                        self.tanks[j].collision_vx = -nx * max_v
                        self.tanks[j].collision_vy = -nx * max_v
                        self.tanks[j].collision_frame = COLLISION_BOUNCE_DECCELLERATE_FRAMES
                        self.tanks[j].recv_damage(self.tanks[i].body_damage, unit_i_hp_before_hit, self.tanks[i].type)

    def step(self, actions=None):
        self.step_count += 1
        rewards = {i: 0.0 for i in range(self.n_tanks)}
        dones = {i: False for i in range(self.n_tanks)}
        infos = {}

        if actions is None:
            actions = {}
            if self.tanks[0].alive:
                actions[0] = self._get_player_input()
            if self.n_tanks > 1 and self.tanks[1].alive:
                actions[1] = self._get_random_input()

        for i, action in actions.items():
            if self.tanks[i].alive:
                self.tanks[i].update_counter()
                dx, dy, _ = action
                self.tanks[i].update(dx, dy)
                rewards[i] += 0.01

        self._handle_collisions()

        for i in range(self.n_tanks):
            if not self.tanks[i].alive:
                dones[i] = True
                rewards[i] -= 10.0
        if self.step_count >= self.max_steps or sum(unit.alive for unit in self.tanks) <= 1:
            dones = {i: True for i in range(self.n_tanks)}

        obs = {i: self._get_obs(i) for i in range(self.n_tanks) if self.tanks[i].alive}
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
        self.screen.fill((255, 255, 255))

        # Render only agent 0's perspective
        if self.tanks[0].alive:
            surface = self._render_frame(0)
            self.screen.blit(surface, (0, 0))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    env = DiepIOEnvBasic(n_tanks=2, render_mode=True)
    obs, _ = env.reset()
    while True:
        obs, rewards, dones, _, _ = env.step()
        if all(dones.values()):
            break
    env.close()