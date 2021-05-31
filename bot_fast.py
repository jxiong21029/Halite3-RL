import random

import numpy as np

from collections import defaultdict
from engine import Game
from entity import Position, MoveCommand, SpawnShipCommand


class FastBot:
    def __init__(self, player_id, eps=0.0):
        self.id = player_id
        self.shipyard = None
        self.halite = 0
        self.returning = defaultdict(bool)  # ship id: bool
        self.map_starting_halite = None
        self.eps = eps

    def generate_commands(self, game):
        if self.shipyard is None:
            for shipyard in game.constructs.values():
                if shipyard.owner_id == self.id:
                    self.shipyard = shipyard
                    break
            self.map_starting_halite = np.sum(game.cells[:, :, 0])
            
        def dist(a, b):
            return min(abs(a.x - b.x), game.size - abs(a.x - b.x)) + min(abs(a.y - b.y), game.size - abs(a.y - b.y))
        commands = {}
        next_pos = {}  # ship id: next position
        next_ships = []  # next_ships[y][x] = list of ships that will end want to move to y x
        turns_left = game.max_turns - game.turn
        for y in range(game.size):
            next_ships.append([])
            for x in range(game.size):
                next_ships[y].append([])

        self.halite = game.banks[self.id]
        for ship in game.ships.values():
            if ship.owner_id == self.id:
                if (
                        ship.halite > 950
                        or game.turn + dist(ship, self.shipyard) + min(20, (game.max_turns // 10) + 1) > game.max_turns
                ):
                    self.returning[ship.id] = True
                elif ship.x == self.shipyard.x and ship.y == self.shipyard.y:
                    self.returning[ship.id] = False
                if ship.halite < game.cells[ship.y][ship.x][0] // 10:
                    next_pos[ship.id] = Position(ship.x, ship.y)
                else:
                    if random.random() < self.eps:
                        target = Position(random.randint(0, game.size), random.randint(0, game.size))
                    elif self.returning[ship.id]:
                        target = Position(self.shipyard.x, self.shipyard.y)
                    else:
                        target = Position(ship.x, ship.y)
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                p = Position((ship.x + dx) % game.size, (ship.y + dy) % game.size)
                                if (game.cells[p.y][p.x][0]) / (dist(ship, p) + 1) > game.cells[target.y][target.x][0] / (dist(target, ship) + 1):
                                    target = p
                    xdl = (ship.x - target.x) % game.size
                    xdr = (target.x - ship.x) % game.size
                    ydd = (ship.y - target.y) % game.size
                    ydu = (target.y - ship.y) % game.size

                    if xdl == xdr == 0:
                        x_dir = 0
                    elif xdl <= xdr:
                        x_dir = -1
                    else:
                        x_dir = 1

                    if ydd == ydu == 0:
                        y_dir = 0
                    elif ydd <= ydu:
                        y_dir = -1
                    else:
                        y_dir = 1

                    if x_dir != 0 and y_dir != 0:
                        x_pen = game.cells[ship.y][(ship.x + x_dir) % game.size][0]
                        y_pen = game.cells[(ship.y + y_dir) % game.size][ship.x][0]
                        if len(next_ships[ship.y][(ship.x + x_dir) % game.size]) > 0:
                            x_pen += 3000
                        elif game.cells[ship.y][(ship.x + x_dir) % game.size][2] != -1:
                            x_pen += 300
                        if len(next_ships[(ship.y + y_dir) % game.size][ship.x]) > 0:
                            y_pen += 3000
                        elif game.cells[(ship.y + y_dir) % game.size][ship.x][2] != -1:
                            y_pen += 300
                        if x_pen < y_pen:
                            next_pos[ship.id] = Position((ship.x + x_dir) % game.size, ship.y)
                            if x_dir == -1:
                                commands[ship.id] = MoveCommand(ship.id, 'W')
                            else:
                                commands[ship.id] = MoveCommand(ship.id, 'E')
                        else:
                            next_pos[ship.id] = Position(ship.x, (ship.y + y_dir) % game.size)
                            if y_dir == -1:
                                commands[ship.id] = MoveCommand(ship.id, 'S')
                            else:
                                commands[ship.id] = MoveCommand(ship.id, 'N')
                    elif x_dir != 0:
                        next_pos[ship.id] = Position((ship.x + x_dir) % game.size, ship.y)
                        if x_dir == -1:
                            commands[ship.id] = MoveCommand(ship.id, 'W')
                        else:
                            commands[ship.id] = MoveCommand(ship.id, 'E')
                    elif y_dir != 0:
                        next_pos[ship.id] = Position(ship.x, (ship.y + y_dir) % game.size)
                        if y_dir == -1:
                            commands[ship.id] = MoveCommand(ship.id, 'S')
                        else:
                            commands[ship.id] = MoveCommand(ship.id, 'N')
                    else:
                        next_pos[ship.id] = Position(ship.x, ship.y)
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].append(ship)

        q = [
            ship
            for ship in game.ships.values()
            if (
                    ship.owner_id == self.id
                    and ship.id in next_pos
                    and (next_pos[ship.id].x != ship.x or next_pos[ship.id].y != ship.y)
            )
        ]
        while q:
            ship = q.pop()
            nx = next_pos[ship.id].x
            ny = next_pos[ship.id].y
            if len(next_ships[ny][nx]) > 1 and not (turns_left <= 50 and nx == self.shipyard.x and ny == self.shipyard.y):
                cur = Position(ship.x, ship.y)
                done = False
                visited = set()
                while not done:
                    cur = next_pos[game.cells[cur.y][cur.x][2]]
                    # if hits empty or enemy, then not a cycle
                    if game.cells[cur.y][cur.x][2] == -1 or game.ships[game.cells[cur.y][cur.x][2]].owner_id != self.id:
                        break
                    # if ship stops, then not a cycle
                    if cur == next_pos[game.cells[cur.y][cur.x][2]]:
                        break
                    if cur == Position(ship.x, ship.y):
                        done = True
                        continue
                    elif game.cells[cur.y][cur.x][2] in visited:
                        break
                    visited.add(game.cells[cur.y][cur.x][2])
                else:
                    continue
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].remove(ship)
                next_pos[ship.id].x = ship.x
                next_pos[ship.id].y = ship.y
                commands[ship.id] = MoveCommand(ship.id, 'O')
                q.extend(next_ships[ship.y][ship.x])
                next_ships[ship.y][ship.x].append(ship)

        ret = list(commands.values())
        if (len(next_ships[self.shipyard.y][self.shipyard.x]) == 0
                and self.halite >= 1000
                and turns_left > min(100, game.max_turns // 2)
                and np.sum(game.cells[:, :, 0]) * 3 > self.map_starting_halite):
            ret.append(SpawnShipCommand(None))
        return ret

    def __repr__(self):
        return f'FastBot(eps={self.eps})'


if __name__ == '__main__':
    g = Game(num_players=2, size=64, create_replay=True)
    bot0, bot1 = FastBot(player_id=0, eps=0), FastBot(player_id=1, eps=0.1)
    for turn in range(g.max_turns):
        g.step([bot0.generate_commands(g), bot1.generate_commands(g)])
