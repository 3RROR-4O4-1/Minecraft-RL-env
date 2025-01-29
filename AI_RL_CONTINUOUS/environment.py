import time
import gym
from gym import spaces
from javascript import require, On, off
from simple_chalk import chalk
import numpy as np

mineflayer = require("mineflayer")
vec3 = require("vec3")
Vec3 = require("vec3").Vec3

def load_world_from_file(filename):
    world = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                world[key] = value
        print("World file loaded successfully.")
    except FileNotFoundError:
        print("World file not found; continuing without it.")
    return world

world = load_world_from_file('./data/epr.txt')


class MinecraftParkourEnv(gym.Env):
    """
    Extended environment:
      action = [deltaYaw, deltaPitch, forward, left, right, jump].
    """

    def __init__(
            self,
            server_host="localhost",
            server_port=3000,
            view_xz_radius=7,
            view_y_up=3,
            view_y_down=3,
            sleep_after_action=0.2
    ):
        super(MinecraftParkourEnv, self).__init__()

        self.server_host = server_host
        self.server_port = server_port
        self.view_xz_radius = view_xz_radius
        self.view_y_up = view_y_up
        self.view_y_down = view_y_down
        self.sleep_after_action = sleep_after_action

        self.bot = None
        self.bot_connected = False


        # for reward function
        self.deaths = 0
        self.nbjumps = 0
        self.nbsprints = 0



        # We now have 6 action dimensions:
        #   [Δyaw, Δpitch, forward, left, right, jump]
        # Let's define them as follows:
        #   - first two are in [-0.3, 0.3]
        #   - the remaining four are in [-1, 1] which we will threshold in step().
        low  = np.array([-0.3, -0.3, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([ 0.3,  0.3,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        # Observation space remains the same, or adapt as needed
        num_blocks = (2 * self.view_xz_radius + 1) * (self.view_y_up + self.view_y_down + 1) * (
            2 * self.view_xz_radius + 1
        )
        self.observation_space = spaces.Box(
            low=0,  # or -inf if you prefer
            high=255,
            shape=(num_blocks + 6,),
            dtype=float
        )

    def _connect_bot(self):
        if self.bot_connected and self.bot:
            return

        self.bot = mineflayer.createBot({
            "host": self.server_host,
            "port": self.server_port,
            "username": "Jerry",
            "hideErrors": False,
            "viewDistance": 1,
        })

        @On(self.bot, "login")
        def on_login(this):
            print(chalk.green(f"[{self.bot.username}] Logged in!"))

        @On(self.bot, "spawn")
        def on_spawn(this):
            self.deaths += 1
            self.bot_connected = True

        @On(self.bot, "end")
        def on_end(this, reason):
            print(chalk.red(f"Bot disconnected: {reason}"))
            off(self.bot, "login", on_login)
            off(self.bot, "spawn", on_spawn)
            off(self.bot, "end", on_end)
            self.bot_connected = False

    def reset(self, **kwargs):

        self.deaths = 0
        self.nbsprints=0
        self.nbJumps=0


        self._connect_bot()

        if self.bot_connected:
            self.bot.chat("/tp -7.5 27 31.5")

        time.sleep(1.0)

        if self.bot_connected:
            self.bot.clearControlStates()

        obs = self._gather_observation()
        self.current_step = 0
        return obs

    def step(self, action):
        """
        action: [deltaYaw, deltaPitch, forward, left, right, jump].
        We interpret forward/left/right/jump as booleans if > 0.
        """
        if not self.bot_connected:
            return np.zeros(self.observation_space.shape), 0.0, True, {}

        delta_yaw, delta_pitch = action[0], action[1]

        # interpret forward, left, right, jump
        sprint_cmd  = (action[2] > 0.25)
        forward_cmd = (0.25>= action[2] > -0.5)
        left_cmd    = (action[3] > 0.0)
        right_cmd   = (action[4] > 0.0)
        held_jump_cmd = (action[5] > 0.5)
        partial_jump_cmd    = (0.0 >= action[5] > -0.5)

        # 1) Apply orientation changes
        self.bot.entity.yaw += float(delta_yaw)
        self.bot.entity.pitch += float(delta_pitch)

        # 2) Clear old states to avoid conflicts, then set new movement
        self.bot.clearControlStates()
        if sprint_cmd:
            self.bot.setControlState("sprint", True)
            self.nbsprints += 1
        elif forward_cmd:
            self.bot.setControlState("forward", True)
        if left_cmd:
            self.bot.setControlState("left", True)
        if right_cmd:
            self.bot.setControlState("right", True)
        if held_jump_cmd:
            self.bot.setControlState("jump", True)
            self.nbjumps += 1
        elif partial_jump_cmd:
            self.bot.setControlState("jump", True)
            time.sleep(0.1)
            self.bot.setControlState("jump", False)
            self.nbjumps += 1

        # 3) Wait
        time.sleep(self.sleep_after_action)

        # 4) Next observation
        next_obs = self._gather_observation()

        # 5) Reward
        reward = self._compute_reward(next_obs)

        # 6) Done?
        done = self._check_done(next_obs)

        self.current_step += 1
        return next_obs, reward, done, {}

    def _gather_observation(self):
        if not self.bot_connected:
            return np.zeros(self.observation_space.shape, dtype=float)

        pos = self.bot.entity.position
        center_x = int(pos.x)
        center_y = int(pos.y)
        center_z = int(pos.z)

        blocks_list = []
        for dx in range(-self.view_xz_radius, self.view_xz_radius + 1):
            for dy in range(-self.view_y_down, self.view_y_up + 1):
                for dz in range(-self.view_xz_radius, self.view_xz_radius + 1):
                    key = f"{center_x + dx}/{center_y + dy}/{center_z + dz}"
                    if key in world:
                        try:
                            block_id = int(world[key])
                        except ValueError:
                            block_id = 1000
                    else:
                        block_id = 1000
                    blocks_list.append(block_id)

        yaw = self.bot.entity.yaw
        pitch = self.bot.entity.pitch
        vx = self.bot.entity.velocity.x
        vy = self.bot.entity.velocity.y
        vz = self.bot.entity.velocity.z
        speed = (vx**2 + vy**2 + vz**2)**0.5

        obs_array = np.array(
            blocks_list + [pos.x, pos.y, pos.z, yaw, pitch, speed],
            dtype=float
        )
        return obs_array

    def _compute_reward(self, obs_array):
        # Extreme Run Parkour part 1
        pos_z = obs_array[-4]
        return pos_z**4 - 1000*self.deaths

    def _check_done(self, obs_array):
        # Extreme Run Parkour part 1
        pos_z = obs_array[-4]
        if pos_z > 240:
            return True

        return False

    def render(self, mode="human"):
        pass

    def close(self):
        if self.bot and self.bot_connected:
            self.bot.quit("Closing environment")
            self.bot = None
            self.bot_connected = False
