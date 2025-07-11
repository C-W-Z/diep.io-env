# from ray.rllib.env import BaseEnv
import numpy as np
from gymnasium import spaces, Env
import pygame
import sys
from typing import Union, Any

from .config import config as cfg
from .unit import UnitType, Unit
from .tank import Tank, TST
from .polygon import Polygon
from .collision import CollisionHash
from .bullet import Bullet, BulletPool
from .utils import check_obs_in_space, draw_rectangle

class DiepIOEnvBody(Env):
    metadata = {"name": "diepio_v0"}

    def __init__(self, env_config: dict[str, Any] = {}):
        super(DiepIOEnvBody, self).__init__()
        self.n_tanks = 1
        # self.n_polygons = int(np.floor(self.n_tanks * cfg.N_POLYGON_SCALE))
        self.n_polygons = 12
        self.render_mode = env_config.get("render_mode", False)
        self.max_steps = env_config.get("max_steps", 10000)
        self.skip_frames = env_config.get("skip_frames", 1)
        self.skip_frames_counter = 0

        self.map_size = 40
        self.screen_size = 400
        cfg.POLYGON_V_SCALE      = 0.01
        cfg.POLYGON_ROTATE_SPEED = 0.005

        # Maximum number of polygons and tanks to include in the observation
        self.obs_max_polygons = 3

        # Observation space
        # === Part 1: Self state ===
        low  = [0.0, 0.0, 1.0, -1.0, -1.0, 0.0,   0.0] + [0] * 4
        high = [1.0, 1.0, 1.6,  1.0,  1.0, 1.0, 278.0] + [7] * 4

        # === Part 2: Polygons ===
        polygon_low  = [-1.0, -1.0,                               0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        polygon_high = [ 1.0,  1.0, np.sqrt(2) * self.map_size + 1e-6, 1.6,  1.0,  1.0, 1.0, 1.0, 1.0, 1.0]
        self.polygon_features = len(polygon_low)

        # === Padding sizes ===
        low += polygon_low * self.obs_max_polygons
        high += polygon_high * self.obs_max_polygons

        # === Observation space ===
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(9)

        self.action_map = {
            0: [0, 0],  # NONE
            1: [0, 1],  # W
            2: [-1, 0], # A
            3: [0, -1], # S
            4: [1, 0],  # D
            5: [-1, 1], # W+A
            6: [1, 1],  # W+D
            7: [-1, -1],# S+A
            8: [1, -1]  # S+D
        }

        # Initialize rendering
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size * self.n_tanks, self.screen_size))
            self.clock = pygame.time.Clock()

        self.reset()

    @staticmethod
    def _rand_poly_side():
        return np.random.choice([3, 4, 5], p=cfg.POLYGON_SIDE_PROB_LIST)

    def reset(self, *, seed=None, options=None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
        Unit.reset_id_iter()

        self.step_count = 0

        # Mapping of ID -> unit
        self.all_things: dict[int, Union[Tank, Polygon, Bullet]] = {}

        # Collision registry
        self.colhash = CollisionHash(self.map_size, self.map_size * 4 // 10)

        self.tanks = [
            Tank(
                x=np.random.uniform(cfg.BORDER_SIZE, self.map_size-cfg.BORDER_SIZE),
                y=np.random.uniform(cfg.BORDER_SIZE, self.map_size-cfg.BORDER_SIZE),
                score=0,
            )
            for _ in range(self.n_tanks)
        ]
        self.polygons = [
            Polygon(
                x=np.random.uniform(cfg.BORDER_SIZE, self.map_size - cfg.BORDER_SIZE),
                y=np.random.uniform(cfg.BORDER_SIZE, self.map_size - cfg.BORDER_SIZE),
                side=self._rand_poly_side()
            )
            for _ in range(self.n_polygons)
        ]

        self.prev_tanks_score = 0
        self.no_reward_frames = 0

        # add all objects to the map
        # add all objects to the collision registry

        # TODO: remove separate lists and just use self.everything
        for tank in self.tanks:
            self.all_things[tank.id] = tank
            self.colhash.add(tank.x, tank.y, tank.id)

        for poly in self.polygons:
            self.all_things[poly.id] = poly
            self.colhash.add(poly.x, poly.y, poly.id)

        # Initialize RLlib dictionaries
        self._dones = False
        self._infos = {}

        observations = self._get_obs()
        if self.render_mode:
            self.render()
        return observations, self._infos

    def _get_obs(self):
        # return pygame.surfarray.array3d(self._get_frame(for_render=False))

        agent: Tank = self.tanks[0]

        # 1. Player's own state
        obs = [
            agent.x / self.map_size, agent.y / self.map_size, # Position [0.0, 1.0]
            agent.radius,                                   # [0.5, 1.6]
            agent.total_vx, agent.total_vy,                 # Velocity [-1.0, 1.0]
            # agent.rx, agent.ry,                             # Direction [-1.0, 1.0]
            agent.hp / agent.max_hp,                        # Normalized health [0.0, 1.0]
            agent.hp,                                       # raw HP [0.0, 278.0]
            # agent.level,                                    # Player level [1, 45]
            # agent.skill_points,                             # Available skill points [0, 33]
            # Tank stats [0, 7]
            agent.stats[TST.HealthRegen],                   # Health regeneration level
            agent.stats[TST.MaxHealth],                     # Max health level
            agent.stats[TST.BodyDamage],                    # Body damage level
            # agent.stats[TST.BulletSpeed],                   # Bullet speed level
            # agent.stats[TST.BulletPen],                     # Bullet penetration level
            # agent.stats[TST.BulletDamage],                  # Bullet damage level
            # agent.stats[TST.Reload],                        # Reload level
            agent.stats[TST.Speed],                         # Speed level
        ]

        # Observation range
        # observation_size = agent.observation_size
        # min_x = agent.x - observation_size / 2
        # max_x = agent.x + observation_size / 2
        # min_y = agent.y - observation_size / 2
        # max_y = agent.y + observation_size / 2

        # 2. Collect and sort polygons by distance
        polygon_data = []
        for obj in self.polygons:
            if not obj.alive:
                continue
            dx, dy = obj.x - agent.x, obj.y - agent.y
            distance = np.hypot(dx, dy)
            side_one_hot = [0, 0, 0]
            side_one_hot[obj.side - 3] = 1
            features = [
                np.float32(dx / max(distance, 1e-6)),
                np.float32(dy / max(distance, 1e-6)),
                np.float32(distance),
                np.float32(obj.radius),
                np.float32(obj.total_vx),
                np.float32(obj.total_vy),
                np.float32(obj.hp / obj.max_hp),
            ] + side_one_hot
            polygon_data.append((distance, obj.x, obj.y, features))

        # Sort by distance and select top k
        polygon_data.sort(key=lambda x: x[0])
        selected_polygons = polygon_data[:self.obs_max_polygons]

        # 3. Create polygon observation array
        polygon_obs = np.zeros((self.obs_max_polygons, self.polygon_features), dtype=np.float32)
        for i, (_, x, y, features) in enumerate(selected_polygons):
            # if min_x <= x <= max_x and min_y <= y <= max_y:
            polygon_obs[i, :] = np.array(features, dtype=np.float32)

        # 4. Combine observations
        return np.concatenate([obs, polygon_obs.flatten()], dtype=np.float32)

    def _render_skill_panel(self, tank: Tank, screen, offset_x, offset_y):
        scale = self.screen_size / 800

        # 1. prepare fonts and dynamic header sizes
        font = pygame.font.Font(None, int(24 * scale))
        line_spacing    = font.get_linesize()    # height of one text line
        header_padding  = 5 * scale # top & bottom padding inside header
        header_height   = header_padding + line_spacing * 2 + header_padding * scale

        # 2. prepare skill list & panel dimensions
        skills = [
            ("Health Regen", (0, 255, 0)),
            ("Max Health", (128, 0, 128)),
            ("Body Damage", (255, 0, 255)),
            ("Bullet Speed", (255, 255, 0)),
            ("Bullet Penetration", (0, 0, 255)),
            ("Bullet Damage", (255, 165, 0)),
            ("Reload", (0, 255, 255)),
            ("Movement Speed", (255, 0, 0)),
        ]
        line_height     = 25  * scale  # vertical spacing per skill entry
        progress_bar_h  = 20  * scale
        bottom_padding  = 10  * scale
        panel_width     = 250 * scale

        # compute full panel height
        panel_height = header_height + len(skills) * line_height + progress_bar_h + bottom_padding

        # if panel would go off-screen, shift it up
        screen_h = self.screen_size
        if offset_y + panel_height > screen_h:
            offset_y = max(0, screen_h - panel_height)

        # 3. draw panel background
        pygame.draw.rect(
            screen, (100, 100, 100),
            (offset_x, offset_y, panel_width, panel_height)
        )

        # 4. draw progress bar background
        bar_x = offset_x + 10 * scale
        bar_y = offset_y + panel_height - progress_bar_h - bottom_padding
        bar_w = panel_width - 20 * scale
        pygame.draw.rect(screen, (0, 0, 0), (bar_x, bar_y, bar_w, progress_bar_h))

        # compute progress based on current level range
        exp_list = cfg.EXP_LIST
        curr_level = tank.level
        curr_exp = tank.score
        curr_threshold = exp_list[curr_level]
        if curr_level + 1 < len(exp_list):
            next_threshold = exp_list[curr_level + 1]
            exp_range = next_threshold - curr_threshold
            # avoid division by zero
            progress = (curr_exp - curr_threshold) / exp_range if exp_range > 0 else 1.0
        else:
            progress = 1.0
        # clamp between 0 and 1
        progress = max(0.0, min(progress, 1.0))

        # draw filled portion
        pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_w * progress), progress_bar_h))

        # 5. render header text at top
        # Skill Points
        sp_surf = font.render(f"Skill Points: {tank.skill_points}", True, (255,255,255))
        screen.blit(sp_surf, (offset_x + 10 * scale, offset_y + header_padding))
        # Score & Level
        lvl_y   = offset_y + header_padding + line_spacing
        lvl_surf = font.render(f"Score: {tank.score}  Level: {tank.level} Tank", True, (255,255,255))
        screen.blit(lvl_surf, (offset_x + 10 * scale, lvl_y))

        # 6. render each skill entry
        self.skill_buttons.clear()
        for i, (skill_name, _) in enumerate(skills):
            y = offset_y + header_height + i * line_height
            level = tank.stats[TST(i)]
            text_surf = font.render(f"{skill_name}: {level}", True, (255,255,255))
            screen.blit(text_surf, (offset_x + 10 * scale, y))

            # draw plus button
            btn = pygame.Rect(offset_x + 200 * scale, y - 5 * scale, 30 * scale, 20 * scale)
            btn_color = (0,200,0) if tank.skill_points > 0 else (150,150,150)
            pygame.draw.rect(screen, btn_color, btn)
            plus_surf = font.render("+", True, (0,0,0))
            screen.blit(plus_surf, (btn.x + 8 * scale, btn.y + 2))

            self.skill_buttons.append((btn, i))

    def _get_frame(self, for_render=False):
        # Create a SCREEN_SIZE x SCREEN_SIZE surface
        surface = pygame.Surface((self.screen_size, self.screen_size))
        surface.fill("#eeeeee")  # White background

        # Center on the agent
        agent = self.tanks[0]
        center_x, center_y = agent.x, agent.y
        observation_size = agent.observation_size

        # Calculate scale: SCREEN_SIZE pixels cover observation_size map units
        grid_size = self.screen_size / observation_size  # e.g., 1000 / 40 = 25
        screen_half = self.screen_size // 2  # 500

        # Draw black rectangles for areas outside map boundaries
        left_boundary = int(screen_half + (cfg.BORDER_SIZE - center_x) * grid_size)  # x = BOARDER_SIZE
        right_boundary = int(screen_half + (self.map_size - cfg.BORDER_SIZE - center_x) * grid_size)  # x = MAP_SIZE - BOARDER_SIZE
        top_boundary = int(screen_half + (cfg.BORDER_SIZE - center_y) * grid_size)  # y = BOARDER_SIZE
        bottom_boundary = int(screen_half + (self.map_size - cfg.BORDER_SIZE - center_y) * grid_size)  # y = MAP_SIZE - BOARDER_SIZE

        black_color = (191, 191, 191)
        if left_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, left_boundary, self.screen_size))
        if right_boundary < self.screen_size:
            pygame.draw.rect(surface, black_color, (right_boundary, 0, self.screen_size - right_boundary, self.screen_size))
        if top_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, self.screen_size, top_boundary))
        if bottom_boundary < self.screen_size:
            pygame.draw.rect(surface, black_color, (0, bottom_boundary, self.screen_size, self.screen_size - bottom_boundary))

        # Draw grid lines (spacing = 1 map units)
        if for_render:
            grid_color = (150, 150, 150)  # Light gray
            # Calculate visible grid lines
            min_x = max(0, center_x - observation_size / 2)
            max_x = min(self.map_size, center_x + observation_size / 2)
            min_y = max(0, center_y - observation_size / 2)
            max_y = min(self.map_size, center_y + observation_size / 2)
            # Grid line positions
            x_grid = np.arange(np.ceil(min_x), np.floor(max_x) + 1)
            y_grid = np.arange(np.ceil(min_y), np.floor(max_y) + 1)

            left_boundary = int(screen_half + (0 - center_x) * grid_size)  # x = 0
            right_boundary = int(screen_half + (self.map_size - center_x) * grid_size)  # x = MAP_SIZE
            top_boundary = int(screen_half + (0 - center_y) * grid_size)  # y = 0
            bottom_boundary = int(screen_half + (self.map_size - center_y) * grid_size)  # y = MAP_SIZE

            # Draw vertical lines
            for x in x_grid:
                pixel_x = int(screen_half + (x - center_x) * grid_size)
                if 0 <= pixel_x < self.screen_size:
                    pygame.draw.line(surface, grid_color, (pixel_x, top_boundary), (pixel_x, bottom_boundary), 1)
            # Draw horizontal lines
            for y in y_grid:
                pixel_y = int(screen_half + (y - center_y) * grid_size)
                if 0 <= pixel_y < self.screen_size:
                    pygame.draw.line(surface, grid_color, (left_boundary, pixel_y), (right_boundary, pixel_y), 1)

        # Draw all polygons
        for unit in self.polygons:
            if not unit.alive:
                continue
            rel_x = unit.x - center_x
            rel_y = unit.y - center_y
            pixel_x = int(screen_half + rel_x * grid_size)
            pixel_y = int(screen_half + rel_y * grid_size)

            # Determine color and side length based on polygon side count
            n_sides = int(round(unit.side))
            if n_sides == 3:  # Triangle (red)
                color = "#ff4000"
                radius = cfg.POLYGON_RADIUS[3] * grid_size # Scale radius to screen
            elif n_sides == 4:  # Square (yellow)
                color = "#ffff00"
                radius = cfg.POLYGON_RADIUS[4] * grid_size
            elif n_sides == 5:  # Pentagon (dark blue)
                color = "#5010aa"
                radius = cfg.POLYGON_RADIUS[5] * grid_size
            else:
                continue  # Skip invalid polygons

            # Calculate polygon vertices
            vertices = []
            for i in range(n_sides):
                # Default angle for regular polygon, rotated by unit.angle
                theta = (2 * np.pi * i / n_sides) + unit.angle
                vx = radius * np.cos(theta)
                vy = radius * np.sin(theta)
                # Translate to polygon center
                vertices.append((pixel_x + vx, pixel_y - vy))  # Invert y for Pygame

            # Calculate bounding box of the polygon
            vertices_x = [v[0] for v in vertices]
            vertices_y = [v[1] for v in vertices]
            min_x = min(vertices_x)
            max_x = max(vertices_x)
            min_y = min(vertices_y)
            max_y = max(vertices_y)

            # Check if the polygon's bounding box intersects with the screen
            if (max_x < 0 or min_x >= self.screen_size or
                max_y < 0 or min_y >= self.screen_size):
                continue  # Polygon is completely outside the screen

            # Collision Box
            # pygame.draw.circle(surface, (127, 127, 127), (pixel_x, pixel_y), int(unit.radius * grid_size))

            if unit.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
                color = (255, 255, 255)

            # Draw polygon
            pygame.draw.polygon(surface, color, vertices)

            # Draw HP bar if the polygon's center is on-screen or near the edge
            # if not (-grid_size <= pixel_x <= self.screen_size + grid_size and -grid_size <= pixel_y <= self.screen_size + grid_size):
            #     continue

            if unit.hp < unit.max_hp:
                half_unit_size = grid_size * unit.radius
                pygame.draw.rect(
                    surface, (0, 0, 0),
                    (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2, max(1, self.screen_size / 200))
                )
                pygame.draw.rect(
                    surface, (0, 216, 0),
                    (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2 * unit.hp / unit.max_hp, max(1, self.screen_size / 200))
                )

        # Draw all tanks
        for i, unit in enumerate(self.tanks):
            if not unit.alive:
                continue
            rel_x = unit.x - center_x
            rel_y = unit.y - center_y
            pixel_x = int(screen_half + rel_x * grid_size)
            pixel_y = int(screen_half + rel_y * grid_size)

            min_x = pixel_x - unit.radius * 2
            max_x = pixel_x + unit.radius * 2
            min_y = pixel_y - unit.radius * 2
            max_y = pixel_y + unit.radius * 2

            # Check if the unit's bounding box intersects with the screen
            if (max_x < 0 or min_x >= self.screen_size or
                max_y < 0 or min_y >= self.screen_size):
                continue  # unit is completely outside the screen

            # Draw orientation line
            barrel_x = pixel_x + unit.radius * grid_size * unit.rx
            barrel_y = pixel_y - unit.radius * grid_size * unit.ry
            draw_rectangle(
                surface, barrel_x, barrel_y,
                2 * unit.radius * grid_size, grid_size,
                "#999999",
                unit.rx, unit.ry
            )

            # Draw body
            color = (0, 127, 255) if unit.id == agent.id else (255, 0, 0)
            if unit.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
                color = (255, 255, 255)
            pygame.draw.circle(
                surface, color,
                (pixel_x, pixel_y),
                int(unit.radius * grid_size)
            )
            # Draw HP bar
            if unit.hp < unit.max_hp:
                half_unit_size = grid_size * unit.radius
                pygame.draw.rect(
                    surface, (0, 0, 0),
                    (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2, max(1, self.screen_size / 200))
                )
                pygame.draw.rect(
                    surface, (0, 216, 0),
                    (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2 * unit.hp / unit.max_hp, max(1, self.screen_size / 200))
                )

        # for bullet in self.bullet_pool.bullets:
        #     if not bullet.alive:
        #         continue
        #     rel_x = bullet.x - center_x
        #     rel_y = bullet.y - center_y
        #     pixel_x = int(screen_half + rel_x * grid_size)
        #     pixel_y = int(screen_half + rel_y * grid_size)
        #     if 0 <= pixel_x < self.screen_size and 0 <= pixel_y < self.screen_size:
        #         color = (0, max(0, 127 - bullet.move_frame), max(0, 255 - bullet.move_frame)) if bullet.tank.id == agent_id else (255, 0, 0)
        #         # color = (0, 127, 255) if bullet.tank.id == 0 else (255, 0, 0)
        #         if bullet.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
        #             color = (255, 255, 255)
        #         pygame.draw.circle(surface, color, (pixel_x, pixel_y), int(bullet.radius * grid_size))

        #         # Draw HP bar
        #         if for_render and bullet.hp < bullet.max_hp:
        #             half_unit_size = grid_size * bullet.radius
        #             pygame.draw.rect(
        #                 surface, (0, 0, 0),
        #                 (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2, max(1, self.screen_size / 200))
        #             )
        #             pygame.draw.rect(
        #                 surface, (0, 216, 0),
        #                 (pixel_x - half_unit_size, pixel_y + half_unit_size * 1.3, half_unit_size * 2 * bullet.hp / bullet.max_hp, max(1, self.screen_size / 200))
        #             )

        if for_render:
            # Render skill panel for agent 0
            self.skill_buttons = []  # Clear previous buttons
            scale = self.screen_size / 800
            self._render_skill_panel(self.tanks[0], surface, 10 * scale, self.screen_size - 260 * scale)  # Left-bottom position

        return surface
        # Convert surface to RGB NumPy array
        # obs_array = pygame.surfarray.array3d(obs_surface)  # Shape: (500, 500, 3)
        # return obs_array

    def _auto_choose_skill(self, mode=0):
        tank: Tank = self.tanks[0]

        if tank.skill_points == 0:
            return 0

        if mode == 1: # bullet shoot
            min_skill = min(tank.stats[TST.BulletPen], tank.stats[TST.BulletDamage], tank.stats[TST.Reload])
            if min_skill == 7 and tank.stats[TST.BulletSpeed] < 5:
                return TST.BulletSpeed + 1
            if min_skill == 7 and tank.stats[TST.BulletSpeed] >= 5:
                return np.random.choice([TST.HealthRegen, TST.MaxHealth, TST.BulletSpeed, TST.Speed, TST.BodyDamage]) + 1
            for i in [TST.BulletPen, TST.BulletDamage, TST.Reload]:
                if tank.stats[i] == min_skill:
                    return i + 1

        elif mode == 2: # body damage
            min_skill = min(tank.stats[TST.HealthRegen], tank.stats[TST.MaxHealth], tank.stats[TST.BodyDamage])
            if min_skill < 7:
                return np.random.choice([TST.HealthRegen, TST.MaxHealth, TST.BodyDamage]) + 1
            if min_skill == 7 and tank.stats[TST.Speed] < 7:
                return TST.Speed + 1
            if min_skill == 7 and tank.stats[TST.Speed] == 7:
                return 0
            # for i in [TST.HealthRegen, TST.MaxHealth, TST.BodyDamage]:
            #     if tank.stats[i] == min_skill:
            #         return i + 1

        return np.random.randint(1, 9) # choose 1 ~ 8

    def _auto_shoot(self):
        tank: Tank = self.tanks[0]
        self_x, self_y = tank.x, tank.y

        min_x = tank.x - tank.observation_size / 2
        max_x = tank.x + tank.observation_size / 2
        min_y = tank.y - tank.observation_size / 2
        max_y = tank.y + tank.observation_size / 2

        best_rx, best_ry = 0, 0
        min_distance = tank.observation_size * 10

        for unit in self.polygons:
            if not (min_x <= unit.x <= max_x and min_y <= unit.y <= max_y):
                continue
            rx, ry = unit.x - self_x, unit.y - self_y
            distance = np.hypot(rx, ry)
            if distance != 0 and distance < min_distance:
                best_rx, best_ry = rx / distance, ry / distance
                min_distance = distance

        for unit in self.tanks:
            if unit.id == tank.id:
                continue
            if not (min_x <= unit.x <= max_x and min_y <= unit.y <= max_y):
                continue
            rx, ry = unit.x - self_x, unit.y - self_y
            distance = np.hypot(rx, ry)
            if distance != 0 and distance < min_distance:
                best_rx, best_ry = rx / distance, ry / distance
                min_distance = distance

        if min_distance == tank.observation_size * 10:
            return 0, 0, 0

        return best_rx, -best_ry, 1

    def _get_player_input(self):
        dx, dy, shoot = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: dy -= 1
        if keys[pygame.K_s]: dy += 1
        if keys[pygame.K_a]: dx -= 1
        if keys[pygame.K_d]: dx += 1
        dx_idx = int(dx + 1)
        dy_idx = int(dy + 1)

        # Update tank direction based on mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen_half = self.screen_size // 2
        rx, ry = mouse_x - screen_half, screen_half - mouse_y
        if rx != 0 or ry != 0:
            magnitude = np.hypot(rx, ry)
            rx, ry = rx / magnitude, ry / magnitude

        # Check for left mouse button click
        if pygame.mouse.get_pressed()[0]:  # Left button is index 0
            shoot = 1

        skill_index = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            skill_index = self._handle_mouse_click(event)
            if skill_index > 0:
                break
            # Handle number key presses for adding skill points
            if event.type == pygame.KEYDOWN:
                # map pygame.K_1..pygame.K_8 to skill indices 1..8
                if pygame.K_1 <= event.key <= pygame.K_8:
                    skill_index = event.key - pygame.K_1 + 1  # 1-based
                    break

        dx_dy_map = {
            (1, 1): 0,  # NONE
            (1, 2): 1,  # W
            (0, 1): 2,  # A
            (1, 0): 3,  # S
            (2, 1): 4,  # D
            (0, 2): 5,  # W+A
            (2, 2): 6,  # W+D
            (0, 0): 7,  # S+A
            (2, 0): 8   # S+D
        }

        action = dx_dy_map[(dx_idx, dy_idx)]
        return action

    def _get_random_input(self):
        dx, dy, shoot = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: dy -= 1
        if keys[pygame.K_DOWN]: dy += 1
        if keys[pygame.K_LEFT]: dx -= 1
        if keys[pygame.K_RIGHT]: dx += 1
        dx_idx = int(dx + 1)
        dy_idx = int(dy + 1)

        rx, ry = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        magnitude = np.hypot(rx, ry)
        if magnitude > 0:
            rx, ry = rx / magnitude, ry / magnitude
        else:
            rx, ry = 1.0, 0.0

        shoot = np.random.randint(2)
        skill_index = 0

        dx_dy_map = {
            (1, 1): 0,  # NONE
            (1, 2): 1,  # W
            (0, 1): 2,  # A
            (1, 0): 3,  # S
            (2, 1): 4,  # D
            (0, 2): 5,  # W+A
            (2, 2): 6,  # W+D
            (0, 0): 7,  # S+A
            (2, 0): 8   # S+D
        }

        action = dx_dy_map[(dx_idx, dy_idx)]
        return action

    def _handle_collisions(self):
        # handle collisions for all tanks & polygons
        for i in range(self.n_tanks):
            tank0 = self.tanks[i]
            nearby_id = self.colhash.nearby(tank0.x, tank0.y, tank0.id)

            for thing_id in nearby_id:
                thing = self.all_things[thing_id]
                # if thing.type == UnitType.Tank:
                #     self.__tank_on_tank(tank0, thing)
                if thing.type == UnitType.Polygon:
                    self.__tank_on_polygon(tank0, thing)

        for i in range(self.n_polygons):
            poly0 = self.polygons[i]
            nearby_id = self.colhash.nearby(poly0.x, poly0.y, poly0.id)

            for thing_id in nearby_id:
                thing = self.all_things[thing_id]
                if thing.type == UnitType.Polygon:
                    self.__polygon_on_polygon(poly0, thing)

        # handle collisions between bullets and other objects
        # self._handle_bullet_collisions()

    def _handle_bullet_collisions(self):
        """Handle collisions between bullets and tanks/polygons."""
        for bullet in self.bullet_pool.bullets:
            if not bullet.alive:
                continue

            # query nearby object IDs from spatial hash
            nearby_ids = self.colhash.nearby(bullet.x, bullet.y, bullet.id, bullet_owner_id=bullet.tank.id)

            for oid in nearby_ids:
                thing = self.all_things.get(oid)
                # skip missing or dead
                if thing is None or not thing.alive:
                    continue

                # compute center–center distance
                dx = thing.x - bullet.x
                dy = thing.y - bullet.y
                distance = np.hypot(dx, dy)
                # only collide when circles overlap
                if distance > thing.radius + bullet.radius:
                    continue

                if distance == 0:
                    dx, dy = 1.0, 0.0
                    distance = 1.0

                nx, ny = dx / distance, dy / distance
                thing_hp_before_hit = thing.hp

                # apply damage: bullet → thing, then penetration damage back to bullet

                if thing.invulberable_frame == 0:
                    thing.collision_vx = nx * cfg.BASE_MAX_VELOCITY * cfg.BULLET_COLLIDER_BOUNCE_V_SCALE
                    thing.collision_vy = ny * cfg.BASE_MAX_VELOCITY * cfg.BULLET_COLLIDER_BOUNCE_V_SCALE
                    thing.collision_frame = cfg.BULLET_BOUNCE_DEC_FRAMES
                    thing.max_collision_frame = cfg.BULLET_BOUNCE_DEC_FRAMES
                    bullet.deal_damage(thing)

                    self._rewards += 0.1

                max_v = cfg.BASE_MAX_VELOCITY * cfg.BULLET_BOUNCE_V_SCALE
                # print(max_v)

                bullet.collision_vx = -nx * max_v
                bullet.collision_vy = -nx * max_v
                bullet.collision_frame = cfg.BULLET_BOUNCE_DEC_FRAMES
                bullet.max_collision_frame = cfg.BULLET_BOUNCE_DEC_FRAMES

                if bullet.invulberable_frame == 0:

                    thing.deal_damage(bullet, thing_hp_before_hit)

                # if bullet HP ≤ 0 after penetration, remove it
                if not bullet.alive:
                    del self.all_things[bullet.id]
                # each bullet hits at most one target per frame
                break

    def __tank_on_tank(self, tank0: Tank, tank1: Tank):
        if not (tank0.alive and tank1.alive):
            return

        dx = tank0.x - tank1.x
        dy = tank0.y - tank1.y
        distance = np.hypot(dx, dy)
        if distance > tank0.radius + tank1.radius:
            return

        if distance == 0:
            dx, dy = 1.0, 0.0
            distance = 1.0

        nx, ny = dx / distance, dy / distance
        max_v = cfg.BASE_MAX_VELOCITY * cfg.TANK_TANK_BOUNCE_V_SCALE
        tank0_hp_before_hit = tank0.hp

        tank0_score, tank1_score = tank0.score, tank1.score

        if tank0.invulberable_frame == 0 and tank1.invulberable_frame == 0:
            tank0.collision_vx = nx * max_v
            tank0.collision_vy = ny * max_v
            tank0.collision_frame = cfg.TANK_BOUNCE_DEC_FRAMES
            tank0.max_collision_frame = cfg.TANK_BOUNCE_DEC_FRAMES
            tank1.deal_damage(tank0)

            tank1.collision_vx = -nx * max_v
            tank1.collision_vy = -nx * max_v
            tank1.collision_frame = cfg.TANK_BOUNCE_DEC_FRAMES
            tank1.max_collision_frame = cfg.TANK_BOUNCE_DEC_FRAMES
            tank0.deal_damage(tank1, tank0_hp_before_hit)

            if not tank0.alive and not tank1.alive:
                tank0.reset_score(tank0_score)
                tank1.reset_score(tank1_score)
                return

            if tank0.last_collider_id == tank1.id:
                tank0.same_collider_counter += 1
            else:
                tank0.same_collider_counter = 1
            tank0.last_collider_id = tank1.id
            tank0.same_collider_counter_reset_frame = cfg.TANK_LONG_INVULNERABLE_FRAMES
            if tank1.last_collider_id == tank0.id:
                tank1.same_collider_counter += 1
            else:
                tank1.same_collider_counter = 1
            tank1.last_collider_id = tank0.id
            tank1.same_collider_counter_reset_frame = cfg.TANK_LONG_INVULNERABLE_FRAMES

    def __tank_on_polygon(self, tank: Tank, poly: Polygon):
        if not (tank.alive and poly.alive):
            return

        dx = tank.x - poly.x
        dy = tank.y - poly.y
        distance = np.hypot(dx, dy)
        if distance > tank.radius + poly.radius:
            return

        # avoid zero division
        if distance == 0:
            dx, dy = 1.0, 0.0
            distance = 1.0
        nx, ny = dx / distance, dy / distance
        tank_hp_before_hit = tank.hp


        if tank.invulberable_frame == 0 and poly.invulberable_frame == 0:
            # Bounce and damage tank
            tank.collision_vx    =  nx * cfg.BASE_MAX_VELOCITY * cfg.TANK_BOUNCE_V_SCALE
            tank.collision_vy    =  ny * cfg.BASE_MAX_VELOCITY * cfg.TANK_BOUNCE_V_SCALE
            tank.collision_frame = cfg.POLYGON_TANK_BOUNCE_DEC_FRAMES
            tank.max_collision_frame = cfg.POLYGON_TANK_BOUNCE_DEC_FRAMES
            poly.deal_damage(tank)

            # Bounce and damage polygon
            poly.collision_vx    = -nx * cfg.BASE_MAX_VELOCITY * cfg.POLYGON_TANK_BOUNCE_V_SCALE
            poly.collision_vy    = -ny * cfg.BASE_MAX_VELOCITY * cfg.POLYGON_TANK_BOUNCE_V_SCALE
            poly.collision_frame = cfg.POLYGON_TANK_BOUNCE_DEC_FRAMES
            poly.max_collision_frame = cfg.POLYGON_TANK_BOUNCE_DEC_FRAMES
            tank.deal_damage(poly, tank_hp_before_hit)
            if tank.hp > 25:
                self._rewards += 0.1

            if tank.last_collider_id == poly.id:
                tank.same_collider_counter += 1
            else:
                tank.same_collider_counter = 1
            tank.last_collider_id = poly.id
            tank.same_collider_counter_reset_frame = cfg.TANK_LONG_INVULNERABLE_FRAMES

    def __polygon_on_polygon(self, poly0: Polygon, poly1: Polygon):
        if not (poly0.alive and poly1.alive):
            return

        dx = poly0.x - poly1.x
        dy = poly0.y - poly1.y
        distance = np.hypot(dx, dy)
        if distance > poly0.radius + poly1.radius:
            return

        if distance == 0:
            dx, dy = 1.0, 0.0
            distance = 1.0

        nx, ny = dx / distance, dy / distance
        max_v = cfg.BASE_MAX_VELOCITY * cfg.POLYGON_BOUNCE_V_SCALE

        if poly0.invulberable_frame == 0:
            poly0.collision_vx = nx * max_v
            poly0.collision_vy = ny * max_v
            poly0.collision_frame = cfg.POLYGON_BOUNCE_DEC_FRAMES
            poly0.max_collision_frame = cfg.POLYGON_BOUNCE_DEC_FRAMES

        if poly1.invulberable_frame == 0:
            poly1.collision_vx = -nx * max_v
            poly1.collision_vy = -nx * max_v
            poly1.collision_frame = cfg.POLYGON_BOUNCE_DEC_FRAMES
            poly1.max_collision_frame = cfg.POLYGON_BOUNCE_DEC_FRAMES

    def _handle_mouse_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
            mouse_pos = pygame.mouse.get_pos()
            for button_rect, skill_index in self.skill_buttons:
                if button_rect.collidepoint(mouse_pos):
                    return skill_index + 1
        return 0

    def step(self, actions: dict[str, dict], skip_frame=False) -> tuple[
        dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]
    ]:
        self.step_count += 1
        self._rewards = 0.0
        observations = None
        truncations = False

        # Process actions for each agent
        tank = self.tanks[0]
        if tank.alive and not self._dones:

            tank.regen_health()
            tank.update_counter()

            dx, dy = self.action_map[int(actions)]
            skill_index = self._auto_choose_skill(mode=2)

            # invalid move reward
            if (dx < 0 and tank.x <= tank.radius + 1e-6 or
                dx > 0 and tank.x >= self.map_size - tank.radius - 1e-6 or
                dy < 0 and tank.y <= tank.radius + 1e-6 or
                dy > 0 and tank.y >= self.map_size - tank.radius - 1e-6):
                self._rewards -= 0.01

            old_x, old_y = tank.x, tank.y

            # try to add a point; if successful, update properties
            if skill_index > 0:
                if tank.add_points(skill_index - 1):
                    tank.calc_stats_properties()
                    self._rewards += 1
                else:
                    self._rewards -= 0.002

            # avoid zero-division
            if dx != 0 or dy != 0:
                magnitude = np.hypot(dx, dy)
                dx, dy = dx / magnitude, dy / magnitude

            # avoid zero-division
            # if rx != 0 or ry != 0:
            #     magnitude = np.hypot(rx, ry)
            #     tank.rx, tank.ry = rx / magnitude, ry / magnitude

            tank.move(dx, dy, self.map_size)
            self.colhash.update(old_x, old_y, tank.x, tank.y, tank.id)

            # if shoot > 0.5 and tank.reload_counter <= 0:
            #     new_bullet = self.bullet_pool.get_new_bullet(tank)
            #     self.all_things[new_bullet.id] = new_bullet
            #     tank.reload_counter = tank.reload_frames

            #     tank.recoil_vx += -1 * tank.rx
            #     tank.recoil_vy -= -1 * tank.ry

        for poly in self.polygons:
            old_x, old_y = poly.x, poly.y

            if poly.alive:
                poly.regen_health()
                poly.update_counter()
                poly.update_direction()
                poly.move(poly.rx, poly.ry, self.map_size)
                self.colhash.update(old_x, old_y, poly.x, poly.y, poly.id)
            else:
                # respawn
                poly.__init__(
                    x=np.random.uniform(cfg.BORDER_SIZE, self.map_size - cfg.BORDER_SIZE),
                    y=np.random.uniform(cfg.BORDER_SIZE, self.map_size - cfg.BORDER_SIZE),
                    side=self._rand_poly_side(),
                    new_id=False,
                )
                self.colhash.update(old_x, old_y, poly.x, poly.y, poly.id)
        # Update bullets
        # for bullet in self.bullet_pool.bullets:  # iterate over a copy
        #     if not bullet.alive:
        #         continue
        #     bullet.update_counter()
        #     bullet.move()
        #     if not bullet.alive:
        #         # remove from environment
        #         del self.all_things[bullet.id]

        self._handle_collisions()

        # Compute rewards and observations

        tank = self.tanks[0]
        if not self._dones and not tank.alive:
            tank.hp = 0
            # tank.calc_respawn_score()
            self._rewards -= 10
            self._dones = True

        self._rewards += np.clip(tank.score - self.prev_tanks_score, -500, 500) * 0.1
        self.prev_tanks_score = tank.score

        if self.no_reward_frames > 10 * cfg.FPS:
            self._rewards -= 0.002

        if self._rewards > 0:
            self.no_reward_frames = 0
        else:
            self.no_reward_frames += 1

        if skip_frame:
            observations = None
        else:
            observations = self._get_obs()

        if self.step_count >= self.max_steps or sum(tank.alive for tank in self.tanks) <= self.n_tanks - 1:
            self._dones = True

        if self.render_mode:
            self.render()

        return observations, self._rewards, self._dones, truncations, self._infos

    def render(self):
        if not self.render_mode:
            return

        self.skip_frames_counter += 1
        if self.skip_frames_counter == self.skip_frames:
            self.skip_frames_counter = 0
        else:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill("#eeeeee")

        for i in range(self.n_tanks):
            # Render only agent 0's perspective
            # if self.tanks[0].alive:
            surface = self._get_frame(for_render=True)
            # self.screen.blit(surface, (0, 0))
            self.screen.blit(surface, (self.screen_size * i, 0))

        pygame.display.flip()
        if self.render_mode == "human":
            self.clock.tick(cfg.FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    env_config = {
        "n_tanks": 1,
        "render_mode": "human",
        "max_steps": 5000,
    }

    env = DiepIOEnvBody(env_config)

    obs, _ = env.reset()
    print(env.observation_space)
    print(env.action_space)
    check_obs_in_space(obs, env.observation_space)

    total_rewards = 0

    while True:
        action_0 = env._get_player_input()

        obs, rewards, dones, truncations, infos = env.step(action_0)
        check_obs_in_space(obs, env.observation_space)

        total_rewards += rewards

        if dones:
            break
    env.close()

    print(total_rewards)
