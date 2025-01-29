from javascript import require, On, Once, off
import numpy as np
import math


mineflayer = require('mineflayer')
vec3 = require("vec3")
Vec3 = require('vec3').Vec3

# loading the world as a dictionary makes it much faster that getting the block from the server
def load_world_from_file(filename):
    world = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            world[key] = value
    return world

# world data
world = load_world_from_file('./data/epr.txt')
print("world loaded")

class ParkourEnv:
    def __init__(self, username):
        self.bot = mineflayer.createBot({
            'host': 'localhost',
            'port': 3000,
            'username': username,
            'hideErrors': False,
            
        })
        self.start_position = None

        self.view_xz_radius = 7,
        self.view_y_up = 3,
        self.view_y_down = 3,
        

        self.total_reward = 0
        self.steps = 0
        self.max_steps = 1000
        self._connect_bot()

    def _connect_bot(self):
        @Once(self.bot, 'spawn')
        def on_spawn(*args):
            self.start_position = self.bot.entity.position.clone()
            self.last_position = self.start_position.clone()

        @On(self.bot, 'move')
        def on_move(*args):

            current_pos = self.bot.entity.position
            self.steps += 1

            # Calculate distance reward
            dx = current_pos.x - self.last_position.x
            dz = current_pos.z - self.last_position.z
            distance = math.sqrt(dx ** 2 + dz ** 2)
            self.total_reward += distance * 0.1

            self.last_position = current_pos.clone()

        @On(self.bot, 'falling')
        def on_falling(*args):
            self.total_reward -= 5  # Penalize falling

    def get_state(self):
        """Return bot's current state as observation"""

        pos = self.bot.entity.position

        blocks_list = []
        for dx in range(-self.view_xz_radius, self.view_xz_radius + 1):
            for dy in range(-self.view_y_down, self.view_y_up + 1):
                for dz in range(-self.view_xz_radius, self.view_xz_radius + 1):
                    try:
                        blocks_list.append(1)  # world[(f"{center_x + dx}/{center_y + dy}/{center_z + dz}")])
                    except:
                        blocks_list.append(0)  # means block is not loaded

        # yaw is in [-π, π], pitch in [-π/2, π/2], velocity can be derived from bot.entity.velocity
        yaw = self.bot.entity.yaw
        pitch = self.bot.entity.pitch
        vx = self.bot.entity.velocity.x
        vy = self.bot.entity.velocity.y
        vz = self.bot.entity.velocity.z
        speed = (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5

        # Combine blocks_list + position + yaw + speed.
        obs_array = np.array(
            blocks_list + [pos.x, pos.y, pos.z, yaw, pitch, speed],
            dtype=float
        )

        return obs_array


    def reset(self):
        """Reset environment for new episode"""
        self.bot.quit()
        self.__init__(self.bot._client.username)
        return self.get_state()

    def calculate_fitness(self):
        """Calculate final fitness score"""
        distance = self.start_position.distanceTo(self.bot.entity.position)
        return distance + self.total_reward - (self.steps * 0.01)