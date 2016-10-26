from cookbook import Cookbook

import numpy as np

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4

ROOM_W = 6
ROOM_H = 6

class LightWorld(object):
    def __init__(self, config):
        self.n_actions = 5
        self.n_features = 12
        self.cookbook = Cookbook(config.recipes)
        self.random = np.random.RandomState(0)

    def sample_scenario_with_goal(self, goal):
        goal = self.cookbook.index.get(goal)
        def walk():
            x, y = 0, 0
            for c in goal:
                if c == "L":
                    x -= 1
                elif c == "R":
                    x += 1
                elif c == "U":
                    y -= 1
                elif c == "D":
                    y += 1
                yield x, y

        # figure out board size

        l, r, u, d = 0, 0, 0, 0
        for x, y in walk():
            l = min(l, x)
            r = max(r, x)
            u = min(u, y)
            d = max(d, y)

        l -= self.random.randint(2)
        r += self.random.randint(2)
        u -= self.random.randint(2)
        d += self.random.randint(2)

        rooms_x = r - l + 1
        rooms_y = d - u + 1
        init_x = -l
        init_y = -u

        board_w = ROOM_W * rooms_x + 1
        board_h = ROOM_H * rooms_y + 1
        walls = np.zeros((board_w, board_h))
        walls[0::ROOM_W, :] = 1
        walls[:, 0::ROOM_H] = 1
        
        doors = []
        keys = {}

        # create doors

        # necessary
        px, py = 0, 0
        for x, y in walk():
            dx = x - px
            dy = y - py
            cx = ROOM_W * (init_x + px) + ROOM_W / 2
            cy = ROOM_H * (init_y + py) + ROOM_H / 2             
            wx = cx + ROOM_W / 2 * dx
            wy = cy + ROOM_H / 2 * dy
            kx = cx + self.random.randint(ROOM_W / 2 + 1) - 1
            ky = cy + self.random.randint(ROOM_H / 2 + 1) - 1
            walls[wx, wy] = 0
            doors.append((wx, wy))
            if self.random.rand() < 0.5:
                keys[(kx, ky)] = (wx, wy)
            px, py = x, y

        # unnecessary
        for i_room in range(min(rooms_x, rooms_y)):
            if rooms_x == 1 or rooms_y == 1:
                continue
            px = self.random.randint(rooms_x-1)
            py = self.random.randint(rooms_y-1)
            dx, dy = (1, 0) if self.random.randint(2) else (0, 1)
            cx = ROOM_W * px + ROOM_W / 2
            cy = ROOM_H * py + ROOM_H / 2
            wx = cx + ROOM_W / 2 * dx
            wy = cy + ROOM_H / 2 * dy
            if (wx, wy) in doors:
                continue
            kx = cx + self.random.randint(ROOM_W / 2 + 1) - 1
            ky = cy + self.random.randint(ROOM_H / 2 + 1) - 1
            walls[wx, wy] = 0
            doors.append((wx, wy))
            if self.random.rand() < 0.5:
                keys[(kx, ky)] = (wx, wy)

        # precompute features

        door_features = {d: np.zeros((board_w, board_h, 4)) for d in doors}
        key_features = {k: np.zeros((board_w, board_h, 4)) for k in keys}
        for x in range(board_w):
            for y in range(board_h):
                rx = x / ROOM_W
                ry = y / ROOM_H
                for dx, dy in doors:
                    #if dx / ROOM_W != rx or dy / ROOM_H != ry:
                    #    continue
                    if rx not in ((dx + 1) / ROOM_W, (dx - 1) / ROOM_W):
                        continue
                    if ry not in ((dy + 1) / ROOM_H, (dy - 1) / ROOM_H):
                        continue
                    if (x, y) != (dx, dy) and (x % ROOM_W == 0 or y % ROOM_H == 0):
                        continue
                    strength = 10 - np.sqrt(np.square((x - dx, y - dy)).sum())
                    strength = max(strength, 0)
                    strength /= 10
                    if dx <= x:
                        door_features[dx, dy][x, y, 0] += strength
                    if dx >= x:
                        door_features[dx, dy][x, y, 1] += strength
                    if dy <= y:
                        door_features[dx, dy][x, y, 2] += strength
                    if dy >= y:
                        door_features[dx, dy][x, y, 3] += strength
                for kx, ky in keys:
                    if kx / ROOM_W != rx or ky / ROOM_H != ry:
                        continue
                    if x % ROOM_W == 0 or y % ROOM_H == 0:
                        continue
                    strength = 10 - np.sqrt(np.square((x - kx, y - ky)).sum())
                    strength = max(strength, 0)
                    strength /= 10
                    if kx <= x:
                        key_features[kx, ky][x, y, 0] += strength
                    if kx >= x:
                        key_features[kx, ky][x, y, 1] += strength
                    if ky <= y:
                        key_features[kx, ky][x, y, 2] += strength
                    if ky >= y:
                        key_features[kx, ky][x, y, 3] += strength

        #np.set_printoptions(precision=1)
        #print keys
        #print doors
        #print walls
        #print key_features[keys.keys()[0]][..., 0]
        #print key_features[keys.keys()[0]][..., 1]
        #print door_features[doors[0]][..., 0]
        #print door_features[doors[1]][..., 1]
        #exit()
        init_room = (init_x, init_y)
        gx, gy = list(walk())[-1]
        goal_room = (init_x + gx, init_y + gy)
        return LightScenario(walls, doors, keys, door_features, key_features, 
                init_room, goal_room, self)

class LightScenario(object):
    def __init__(self, walls, doors, keys, door_features, key_features, 
            init_room, goal_room, world):
        self.walls = walls
        self.doors = doors
        self.keys = keys
        self.door_features = door_features
        self.key_features = key_features
        self.init_room = init_room
        self.goal_room = goal_room
        self.world = world

    def init(self):
        ix, iy = self.init_room
        ix = ROOM_W * ix + ROOM_W / 2
        iy = ROOM_H * iy + ROOM_H / 2
        s = LightState(self.walls, self.doors, self.keys, (ix, iy), self)
        return s

class LightState(object):
    def __init__(self, walls, doors, keys, pos, scenario):
        self.walls = walls
        self.doors = doors
        self.keys = keys
        self.pos = pos
        self.scenario = scenario
        self._cached_features = None

    def features(self):
        if self._cached_features is None:
            out = np.zeros(12)
            for door in self.doors:
                df = self.scenario.door_features[door][self.pos[0], self.pos[1], :]
                if door in self.keys.values():
                    out[0:4] += df
                else:
                    out[4:8] += df
            for key in self.keys:
                kf = self.scenario.key_features[key][self.pos[0], self.pos[1], :]
                out[8:12] += kf
            self._cached_features = out
        return self._cached_features

        #return self.scenario.features[self.pos[0], self.pos[1], :]

    def satisfies(self, goal_name, goal_arg):
        px, py = self.pos
        return (px / ROOM_W, py / ROOM_H) == self.scenario.goal_room

    def step(self, action):
        x, y = self.pos
        n_keys = self.keys
        # move actions
        if action == DOWN:
            dx, dy = (0, -1)
        elif action == UP:
            dx, dy = (0, 1)
        elif action == LEFT:
            dx, dy = (-1, 0)
        elif action == RIGHT:
            dx, dy = (1, 0)
        elif action == USE:
            n_keys = dict(self.keys)
            dx, dy = (0, 0)
            if (x, y) in self.keys:
                del n_keys[(x, y)]

        nx, ny = x + dx, y + dy
        if self.walls[nx, ny]:
            nx, ny = x, y
        if (nx, ny) in self.doors and (nx, ny) in self.keys.values():
            nx, ny = x, y
        return 0, LightState(self.walls, self.doors, n_keys, (nx, ny), self.scenario)

    def pp(self):
        w, h = self.walls.shape
        out = ""
        for x in range(w):
            for y in range(h):
                if (x, y) == self.pos and (x, y) in self.keys:
                    out += "%m"
                elif (x, y) == self.pos:
                    out += "%%"
                elif self.walls[x, y]:
                    out += "##"
                elif (x, y) in self.keys:
                    out += "Om"
                elif (x, y) in self.doors and (x, y) in self.keys.values():
                    out += "$$"
                else:
                    out += "  "
            out += "\n"
        out += str(self.satisfies(None, None))
        return out
