import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import sys
from typing import Dict, Union

from config import config as cfg
from unit import UnitType, Unit
from tank import Tank
from polygon import Polygon
from collision import CollisionHash
from bullet import Bullet

class DiepIOEnvBasic(gym.Env):
    def __init__(self, n_tanks=2, render_mode=True):
        super(DiepIOEnvBasic, self).__init__()

        self.n_tanks = n_tanks
        self.n_polygons = int(np.floor(n_tanks * cfg.N_POLYGON_SCALE))
        self.render_mode = render_mode
        self.max_steps = 1000000
        self.bullets: list[Bullet] = []

        # Observation space: Vector of tank states
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_tanks * 6,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32
        )

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((cfg.SCREEN_SIZE, cfg.SCREEN_SIZE))
            self.clock = pygame.time.Clock()
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(cfg.SCREEN_SIZE, cfg.SCREEN_SIZE, 3), dtype=np.uint8
            )

        self.reset()

    @staticmethod
    def _rand_poly_side():
        r = np.random.rand()
        if r < cfg.POLYGON_SIDE_PROB[5]:
            return 5
        elif r < cfg.POLYGON_SIDE_PROB[5] + cfg.POLYGON_SIDE_PROB[4]:
            return 4
        return 3

    def reset(self, seed=None, options=None):
        Unit.reset_id_iter()

        self.bullets: list[Bullet] = []  # List to store all bullets
        self.step_count = 0

        # Mapping of ID -> unit
        self.all_things: Dict[int, Union[Tank, Polygon]] = {}

        # Collision registry
        self.colhash = CollisionHash(cfg.BORDER_SIZE + cfg.MAP_SIZE, cfg.MAP_GRID)

        self.tanks = [
            Tank(
                x=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE-cfg.BORDER_SIZE),
                y=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE-cfg.BORDER_SIZE),
                max_hp=50.0,
                score=0,
            )
            for _ in range(self.n_tanks)
        ]
        self.polygons = [
            Polygon(
                x=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE - cfg.BORDER_SIZE),
                y=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE - cfg.BORDER_SIZE),
                side=self._rand_poly_side()
            )
            for _ in range(self.n_polygons)
        ]

        # add all objects to the map
        # TODO: remove separate lists and just use self.everything
        for tank in self.tanks:
            self.all_things[tank.id] = tank

        for poly in self.polygons:
            self.all_things[poly.id] = poly

        # add all objects to the collision registry
        for tank in self.tanks:
            self.colhash.add(tank.x, tank.y, tank.id)

        for poly in self.polygons:
            self.colhash.add(poly.x, poly.y, poly.id)

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
                obs.extend([tank.x, tank.y, tank.vx, tank.vy, tank.rx, tank.ry])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def _render_frame(self, agent_id):
        # Create a SCREEN_SIZE x SCREEN_SIZE surface
        surface = pygame.Surface((cfg.SCREEN_SIZE, cfg.SCREEN_SIZE))
        surface.fill((255, 255, 255))  # White background

        # Center on the agent
        agent = self.tanks[agent_id]
        center_x, center_y = agent.x, agent.y
        observation_size = agent.observation_size

        # Calculate scale: SCREEN_SIZE pixels cover observation_size map units
        grid_size = cfg.SCREEN_SIZE / observation_size  # e.g., 1000 / 40 = 25
        screen_half = cfg.SCREEN_SIZE // 2  # 500

        # Draw black rectangles for areas outside map boundaries
        left_boundary = int(screen_half + (cfg.BORDER_SIZE - center_x) * grid_size)  # x = BOARDER_SIZE
        right_boundary = int(screen_half + (cfg.MAP_SIZE - cfg.BORDER_SIZE - center_x) * grid_size)  # x = MAP_SIZE - BOARDER_SIZE
        top_boundary = int(screen_half + (cfg.BORDER_SIZE - center_y) * grid_size)  # y = BOARDER_SIZE
        bottom_boundary = int(screen_half + (cfg.MAP_SIZE - cfg.BORDER_SIZE - center_y) * grid_size)  # y = MAP_SIZE - BOARDER_SIZE

        black_color = (191, 191, 191)
        if left_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, left_boundary, cfg.SCREEN_SIZE))
        if right_boundary < cfg.SCREEN_SIZE:
            pygame.draw.rect(surface, black_color, (right_boundary, 0, cfg.SCREEN_SIZE - right_boundary, cfg.SCREEN_SIZE))
        if top_boundary > 0:
            pygame.draw.rect(surface, black_color, (0, 0, cfg.SCREEN_SIZE, top_boundary))
        if bottom_boundary < cfg.SCREEN_SIZE:
            pygame.draw.rect(surface, black_color, (0, bottom_boundary, cfg.SCREEN_SIZE, cfg.SCREEN_SIZE - bottom_boundary))

        # Draw grid lines (spacing = 1 map units)
        grid_color = (150, 150, 150)  # Light gray
        # Calculate visible grid lines
        min_x = max(0, center_x - observation_size / 2)
        max_x = min(cfg.MAP_SIZE, center_x + observation_size / 2)
        min_y = max(0, center_y - observation_size / 2)
        max_y = min(cfg.MAP_SIZE, center_y + observation_size / 2)
        # Grid line positions
        x_grid = np.arange(np.ceil(min_x), np.floor(max_x) + 1)
        y_grid = np.arange(np.ceil(min_y), np.floor(max_y) + 1)

        left_boundary = int(screen_half + (0 - center_x) * grid_size)  # x = 0
        right_boundary = int(screen_half + (cfg.MAP_SIZE - center_x) * grid_size)  # x = MAP_SIZE
        top_boundary = int(screen_half + (0 - center_y) * grid_size)  # y = 0
        bottom_boundary = int(screen_half + (cfg.MAP_SIZE - center_y) * grid_size)  # y = MAP_SIZE

        # Draw vertical lines
        for x in x_grid:
            pixel_x = int(screen_half + (x - center_x) * grid_size)
            if 0 <= pixel_x < cfg.SCREEN_SIZE:
                pygame.draw.line(surface, grid_color, (pixel_x, top_boundary), (pixel_x, bottom_boundary), 1)
        # Draw horizontal lines
        for y in y_grid:
            pixel_y = int(screen_half + (y - center_y) * grid_size)
            if 0 <= pixel_y < cfg.SCREEN_SIZE:
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
                color = (255, 0, 0)
                radius = cfg.POLYGON_RADIUS[3] * grid_size # Scale radius to screen
            elif n_sides == 4:  # Square (yellow)
                color = (255, 255, 0)
                radius = cfg.POLYGON_RADIUS[4] * grid_size
            elif n_sides == 5:  # Pentagon (dark blue)
                color = (0, 0, 139)
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
            if (max_x < 0 or min_x >= cfg.SCREEN_SIZE or
                max_y < 0 or min_y >= cfg.SCREEN_SIZE):
                continue  # Polygon is completely outside the screen

            # Collision Box
            # pygame.draw.circle(surface, (127, 127, 127), (pixel_x, pixel_y), int(unit.radius * grid_size))

            if unit.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
                color = (216, 216, 216)

            # Draw polygon
            pygame.draw.polygon(surface, color, vertices)

            # Draw HP bar if the polygon's center is on-screen or near the edge
            # if not (-grid_size <= pixel_x <= cfg.SCREEN_SIZE + grid_size and -grid_size <= pixel_y <= cfg.SCREEN_SIZE + grid_size):
            #     continue

            if unit.hp < unit.max_hp:
                hp_width = int(grid_size * 2 * unit.radius * unit.hp / unit.max_hp)
                pygame.draw.rect(
                    surface, (0, 216, 0),
                    (pixel_x - grid_size * unit.radius, pixel_y + grid_size * unit.radius, hp_width, 5)
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
            if (max_x < 0 or min_x >= cfg.SCREEN_SIZE or
                max_y < 0 or min_y >= cfg.SCREEN_SIZE):
                continue  # unit is completely outside the screen

            # Draw orientation line
            end_x = pixel_x + unit.radius * 2 * grid_size * unit.rx
            end_y = pixel_y - unit.radius * 2 * grid_size * unit.ry
            pygame.draw.line(
                surface, (127, 127, 127),
                (pixel_x, pixel_y),
                (end_x, end_y),
                int(grid_size)
            )
            # Draw body
            color = (0, 127, 255) if unit.id == 0 else (255, 0, 0)
            if unit.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
                color = (216, 216, 216)
            pygame.draw.circle(
                surface, color,
                (pixel_x, pixel_y),
                int(unit.radius * grid_size)
            )
            # Draw HP bar
            if unit.hp < unit.max_hp:
                hp_width = int(grid_size * 2 * unit.radius * unit.hp / unit.max_hp)
                pygame.draw.rect(
                    surface, (0, 216, 0),
                    (pixel_x - grid_size * unit.radius, pixel_y + grid_size * unit.radius, hp_width, 5)
                )

        for bullet in self.bullets:
            if bullet.alive:
                rel_x = bullet.x - center_x
                rel_y = bullet.y - center_y
                pixel_x = int(screen_half + rel_x * grid_size)
                pixel_y = int(screen_half + rel_y * grid_size)
                if 0 <= pixel_x < cfg.SCREEN_SIZE and 0 <= pixel_y < cfg.SCREEN_SIZE:
                    color = (0, 127, 255) if bullet.tank.id == 0 else (255, 0, 0)
                    if bullet.invulberable_frame >= cfg.INVULNERABLE_FRAMES:
                        color = (216, 216, 216)
                    pygame.draw.circle(surface, color, (pixel_x, pixel_y), int(bullet.radius * grid_size))

                    # Draw HP bar
                    if bullet.hp < bullet.max_hp:
                        hp_width = int(grid_size * 2 * bullet.radius * bullet.hp / bullet.max_hp)  # Scale with grid
                        pygame.draw.rect(
                            surface, (0, 216, 0),
                            (pixel_x - grid_size * bullet.radius, pixel_y + grid_size * bullet.radius, hp_width, 5)
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
            magnitude = np.hypot(dx, dy)
            dx, dy = dx / magnitude, dy / magnitude
        # Update tank direction based on mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen_half = cfg.SCREEN_SIZE // 2
        rx, ry = mouse_x - screen_half, screen_half - mouse_y
        magnitude = np.hypot(rx, ry)
        self.tanks[0].rx, self.tanks[0].ry = rx / magnitude, ry / magnitude
        # Check for left mouse button click
        if pygame.mouse.get_pressed()[0]:  # Left button is index 0
            shoot = 1.0
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
            magnitude = np.hypot(dx, dy)
            dx, dy = dx / magnitude, dy / magnitude
        rx, ry = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        magnitude = np.hypot(rx, ry)
        self.tanks[1].rx, self.tanks[1].ry = rx / magnitude, ry / magnitude
        return np.array([dx, dy, shoot], dtype=np.float32)

    def _handle_collisions(self):
        # handle collisions for all tanks & polygons
        for i in range(self.n_tanks):
            tank0 = self.tanks[i]
            nearby_id = self.colhash.nearby(tank0.x, tank0.y, tank0.id)

            for thing_id in nearby_id:
                thing = self.all_things[thing_id]
                if thing.type == UnitType.Tank:
                    self.__tank_on_tank(tank0, thing)
                elif thing.type == UnitType.Polygon:
                    self.__tank_on_polygon(tank0, thing)

        for i in range(self.n_polygons):
            poly0 = self.polygons[i]
            nearby_id = self.colhash.nearby(poly0.x, poly0.y, poly0.id)

            for thing_id in nearby_id:
                thing = self.all_things[thing_id]
                if thing.type == UnitType.Polygon:
                    self.__polygon_on_polygon(poly0, thing)

        # handle collisions between bullets and other objects
        self._handle_bullet_collisions()

    def _handle_bullet_collisions(self):
        """Handle collisions between bullets and tanks/polygons."""
        for bullet in self.bullets[:]:
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
                    self.bullets.remove(bullet)
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
            tank: Tank = self.tanks[i]
            if tank.alive:
                tank.regen_health()
                tank.update_counter()
                dx, dy, shoot = action
                old_x, old_y = tank.x, tank.y
                tank.move(dx, dy)
                self.colhash.update(old_x, old_y, tank.x, tank.y, tank.id)

                rewards[i] += 0.01

            if shoot > 0.5 and tank.reload_counter <= 0:
                bx = tank.x + tank.radius * tank.rx
                by = tank.y + tank.radius * -tank.ry

                new_bullet = Bullet(
                    x=bx,
                    y=by,
                    max_hp = tank.bullet_max_hp,
                    bullet_damage = tank.bullet_damage,
                    radius = tank.bullet_radius,
                    tank = tank,
                    rx = tank.rx,
                    ry = -tank.ry,
                    v_scale = tank.bullet_v_scale,
                )
                self.bullets.append(new_bullet)
                self.all_things[new_bullet.id] = new_bullet
                tank.reload_counter = tank.reload_frames

        for poly in self.polygons:
            old_x, old_y = poly.x, poly.y
            if poly.alive:
                poly.regen_health()
                poly.update_counter()
                poly.update_direction()
                poly.move(poly.rx, poly.ry)
                self.colhash.update(old_x, old_y, poly.x, poly.y, poly.id)
            else:
                # respawn
                poly.__init__(
                    x=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE - cfg.BORDER_SIZE),
                    y=np.random.uniform(cfg.BORDER_SIZE, cfg.MAP_SIZE - cfg.BORDER_SIZE),
                    side=self._rand_poly_side(),
                    new_id=False,
                )
                self.colhash.update(old_x, old_y, poly.x, poly.y, poly.id)

        # Update bullets
        for bullet in self.bullets[:]:  # iterate over a copy
            bullet.update_counter()
            bullet.move()
            if not bullet.alive:
                # remove from environment
                self.bullets.remove(bullet)
                del self.all_things[bullet.id]

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
        self.clock.tick(cfg.FPS)

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
