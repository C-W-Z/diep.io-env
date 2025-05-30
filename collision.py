# Spatial hashing for collision detection
import numpy as np
from numba import njit

# Tanks and polygons go in here, no bullets please
class CollisionHash:
    def __init__(self, map_size, grid_size):
        self.factor = map_size / grid_size
        self.grid   = np.empty([grid_size, grid_size], dtype=object)
        for i in range(grid_size):
            for j in range(grid_size):
                self.grid[i, j] = set()

    def coord2grid(self, x, y):
        return int(x / self.factor), int(y / self.factor)

    # fetch nearby units
    # Returns: a list of nearby units IDs
    def nearby(self, x, y, ID, bullet_owner_id=None, distance=1):
        sz = self.grid.shape[0]

        i, j = self.coord2grid(x, y)
        nearby_id = []

        for di in range(-distance, distance + 1):
            for dj in range(-distance, distance + 1):
                ci, cj = i + di, j + dj
                if 0 <= ci < sz and 0 <= cj < sz:
                    nearby_id.extend(self.grid[ci, cj])
        try:
            if bullet_owner_id is None:
                nearby_id.remove(ID) # Not a bullet: don't collide with yourself
        except ValueError:
            print(f"[WARN] CollisionHash: Tried to remove {ID} from {nearby_id}")

        if bullet_owner_id in nearby_id:
            nearby_id.remove(bullet_owner_id) # Bullet: don't collide with the tank that shot it

        return nearby_id

    # add a new unit
    def add(self, x, y, ID):
        i, j = self.coord2grid(x, y)
        self._add_direct(i, j, ID)

    def _add_direct(self, i, j, ID):
        self.grid[i, j].add(ID)

    # remove a unit
    def rm(self, x, y, ID):
        i, j = self.coord2grid(x, y)
        self._rm_direct(i, j, ID)

    def _rm_direct(self, i, j, ID):
        if ID in self.grid[i, j]:
            self.grid[i, j].remove(ID)
        else:
            print("[WARN] CollisionHash: missed")

    # update unit position
    def update(self, old_x, old_y, x, y, ID):
        old_i, old_j = self.coord2grid(old_x, old_y)
        i, j = self.coord2grid(x, y)

        if (old_i, old_j) != (i, j):
            self._rm_direct(old_i, old_j, ID)
            self._add_direct(i, j, ID)

    # for debugging
    def where(self, x, y):
        i, j = self.coord2grid(x, y)
        print(f"[DEBUG] Grid location: {i}, {j}")
        print(f"[DEBUG] Content: {self.grid[i, j]}")
