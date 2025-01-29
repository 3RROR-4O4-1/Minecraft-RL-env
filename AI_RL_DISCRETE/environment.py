import time
import gym
from gym import spaces
from javascript import require, On, off
from simple_chalk import chalk
import numpy as np
import itertools

# Mineflayer + vec3
mineflayer = require("mineflayer")
vec3 = require("vec3")
Vec3 = require("vec3").Vec3

# loading the world as a dictionary makes it much faster that getting the block from the server
def load_world_from_file(filename):
    world = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            world[key] = value
    return world

# world data
world = load_world_from_file('./data/spiral.txt')
print("world loaded")


# ============================================
#             GYM ENVIRONMENT                #
# ============================================
class MinecraftParkourEnv(gym.Env):
    """
    A single-bot environment that mimics the OpenAI Gym API:
      - reset() -> returns initial observation
      - step(action) -> applies one action, returns (next_obs, reward, done, info)

    Observations:
      - A list of block IDs around the bot (view size is configurable)
      - The bot's position, yaw, pitch, velocity, etc. (as needed)

    Actions (discrete or multi-discrete):
      - e.g., "forward", "jump", "left", "right", etc.

    Reward & Done:
      - You define how to calculate these (parkour progress, falling off, etc.)
    """

    def __init__(
            self,
            server_host="localhost",
            server_port=3000,
            name="Jerry",
            view_xz_radius=7,
            view_y_up=3,
            view_y_down=10,
            sleep_after_action=0.2,
            yaw_delta=0.1,    # ONLY NEEDED IF YAW HAS TO BE IMPLEMENTED Warning: this will greatly increase the output action space 18 -> 162. 0.1 radian = 5.7 degrees
            pitch_delta=0.1
    ):
        super(MinecraftParkourEnv, self).__init__()

        # Store config
        self.server_host = server_host
        self.server_port = server_port
        self.name = name
        self.view_xz_radius = view_xz_radius
        self.view_y_up = view_y_up
        self.view_y_down = view_y_down
        self.sleep_after_action = sleep_after_action

        # Bot reference
        self.bot = None
        self.bot_connected = False


        # forward_dim  = 3  (0 = no forward/back, 1=forward, 2=back)
        # strafe_dim   = 3  (0 = none, 1=left, 2=right)
        # jump_dim     = 2  (0 = no jump, 1=jump)
        # yaw_dim      = 3  (0 = turn left yaw_delta, 1= no yaw change, 2= turn right yaw_delta)
        # pitch_dim    = 3  (0 = pitch up, 1= no pitch change, 2= pitch down)

        forward_vals = range(3)
        strafe_vals = range(3)
        jump_vals = range(2)
        # yaw_vals = range(3)
        # pitch_vals = range(3)

        # Use itertools.product to enumerate all possible combos
        self.discrete_combos = list(itertools.product(forward_vals, strafe_vals, jump_vals)) #, yaw_vals, pitch_vals)) if needed
        num_actions = len(self.discrete_combos)  # 3*3*2*(3*3) = (18*9)

        # Now we say: self.action_space is Discrete(18 or 162)
        self.action_space = spaces.Discrete(num_actions)

        # Storing  yaw_delta/pitch_delta for applying the increments
        self.yaw_delta = yaw_delta
        self.pitch_delta = pitch_delta


        # Example observation space:
        # We have (2 * view_xz_radius + 1) * (view_y_up + view_y_down + 1) * (2 * view_xz_radius + 1) block IDs,
        # plus continuous data for position/yaw/pitch, etc.
        # For simplicity, let's assume block IDs are integer-coded up to some maximum (e.g. 200).
        # We'll do a shape for the block array of length N, plus some extra dims for pos, yaw, pitch, speed.
        num_blocks = (2 * self.view_xz_radius + 1) * (self.view_y_up + self.view_y_down + 1) * (
                    2 * self.view_xz_radius + 1)
        # E.g., block IDs in [0..255], plus a small continuous vector
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(num_blocks + 4,),  # 4 extra for (x, y, z, yaw, speed), adapt as needed
            dtype=float
        )

    def _connect_bot(self):
        """Create the Mineflayer bot and set up event handlers."""
        if self.bot_connected and self.bot:
            return  # Already connected

        self.bot = mineflayer.createBot({
            "host": self.server_host,
            "port": self.server_port,
            "username": self.name,
            "hideErrors": False,
            "viewDistance": 1,
        })

        @On(self.bot, "login")
        def on_login(this):
            print(chalk.green(f"[{self.bot.username}] Logged in!"))

        @On(self.bot, "spawn")
        def on_spawn(this):
            print(chalk.yellow("Bot spawned!"))
            self.bot_connected = True

        @On(self.bot, "end")
        def on_end(this, reason):
            print(chalk.red(f"Bot disconnected: {reason}"))
            off(self.bot, "login", on_login)
            off(self.bot, "spawn", on_spawn)
            off(self.bot, "end", on_end)
            self.bot_connected = False

    def reset(self, **kwargs):
        """
        Reset the environment:
          1) Connect or reconnect the bot
          2) Teleport to the start of the Exercice
          3) Wait for stable spawn
          4) Return initial observation
          :param **kwargs:
        """
        # Connects the bot if not already connected
        self._connect_bot()

        # Teleport or set a command to go to the start of the map
        # (Adjust to your map's coords)
        if self.bot_connected:
            self.bot.chat("/tp 1 -61 64")

        # Wait a bit for the server to process and load chunks
        time.sleep(1.0)

        # Clear any motion states
        self.bot.clearControlStates()

        # Gather initial observation
        obs = self._gather_observation()

        # Optionally track any internal state (score, steps, etc.)
        self.current_step = 0

        return obs

    def step(self, action):
        """
        Apply exactly one action, wait a short time,
        gather next observation, compute reward, and check if done.
        """
        # action is an integer in [0..17]
        (forward_type, strafe_type, jump_type) = self.discrete_combos[action]
        # forward_type: 0=no forward, 1=forward, 2=back
        # strafe_type: 0=no strafe, 1=left, 2=right
        # jump_type: 0=no jump, 1=jump


        # 1) Apply action
        self._apply_action(forward_type, strafe_type, jump_type)

        # 2) Wait so the bot actually moves
        time.sleep(self.sleep_after_action)

        # 3) Gather next observation
        next_obs = self._gather_observation()

        # 4) Compute reward
        reward = self._compute_reward(next_obs)

        # 5) Check if episode is done
        done = self._check_done(next_obs)

        self.current_step += 1

        # Optionally put any debug info in 'info'
        info = {}

        return next_obs, reward, done, info

    def _gather_observation(self):
        """
        Collect blocks in a region around the bot plus
        position, yaw, pitch, velocity.
        Return as a NumPy array or Python list matching self.observation_space.
        """

        if not self.bot_connected:
            # If somehow we're not connected, return a dummy observation
            return np.zeros(self.observation_space.shape, dtype=float)

        pos = self.bot.entity.position

        blocks_list = []
        for dx in range(-self.view_xz_radius, self.view_xz_radius + 1):
            for dy in range(-self.view_y_down, self.view_y_up + 1):
                for dz in range(-self.view_xz_radius, self.view_xz_radius + 1):
                    try:
                        blocks_list.append(1)       #world[(f"{center_x + dx}/{center_y + dy}/{center_z + dz}")])
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

    def _apply_action(self, forward_type, strafe_type, jump_type):
        """
        Decode the action (0..7) and apply relevant control states.
        For more complex actions, change to multi-discrete or a list of strings.
        """
        if not self.bot_connected:
            return

        # Clear previous states
        self.bot.clearControlStates()

        # forward/back
        if forward_type == 1:
            self.bot.setControlState("forward", True)
        elif forward_type == 2:
            self.bot.setControlState("back", True)

        # left/right
        if strafe_type == 1:
            self.bot.setControlState("left", True)
        elif strafe_type == 2:
            self.bot.setControlState("right", True)

        # jump
        if jump_type == 1:
            self.bot.setControlState("jump", True)

    def _compute_reward(self, obs_array):
        """
        Reward function for parkour:
        - E.g., +0.01 for moving forward on X-axis, +1 if reaching checkpoint, -1 if falling
        - This example just returns 0.0. You must define your own logic.
        """
        # Example: Could compare current x to previous x, or track if y < some threshold => negative reward
        reward = 0.0
        return reward

    def _check_done(self, obs_array):
        """
        Check if the episode is finished:
         - e.g., if the agent falls below a certain Y-level
         - or if it reaches the Goal (currently a certain X coordinate)
        """
        pos_x = obs_array[-6]  # if you appended [pos.x, pos.y, pos.z, yaw, pitch, speed]
        pos_y = obs_array[-5]
        # pos_z = obs_array[-4], etc.

        # For example, done if Y < -100 => fell off
        if pos_y < -100:
            return True

        # Or done if X > some goal
        # if pos_x > 100.0:
        #     return True

        return False

    def close(self):
        """Cleanly disconnect the bot if needed."""
        if self.bot and self.bot_connected:
            self.bot.quit("Closing environment")
            self.bot = None
            self.bot_connected = False
