import time
from javascript import require, On, AsyncTask, off
from simple_chalk import chalk
from code.QUICK_SETUP.action import retrieve_action

# Mineflayer + vec3 from JavaScript world
mineflayer = require("mineflayer")
vec3 = require("vec3")
Vec3 = require("vec3").Vec3

# ============================================
# CONFIG
# ============================================

server_host = "localhost"  # or your server IP
server_port = 3000
reconnect = True
num_bots = 10

def load_world_from_file(filename):
    world = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            world[key] = value
    return world

# world data
world = load_world_from_file('../data/spiral.txt')
print("world loaded")

# ============================================
# BOT
# ============================================

class MCBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        # Bot login args
        self.bot_args = {
            "host": server_host,
            "port": server_port,
            "username": bot_name,
            "hideErrors": False,
            "viewDistance": 1,
        }
        self.reconnect = reconnect
        self.start_bot()
        self.view = []


    def log(self, message):
        print(f"[{self.bot.username}] {message}")

    def start_bot(self):
        self.bot = mineflayer.createBot(self.bot_args)
        self.start_events()

    def start_events(self):
        # --------------------------------------------
        # LOGIN
        # --------------------------------------------
        @On(self.bot, "login")
        def on_login(this):
            self.bot_socket = self.bot._client.socket
            sock_name = (
                self.bot_socket.server if self.bot_socket.server else self.bot_socket._host
            )
            self.log(chalk.green(f"Logged in to {sock_name}"))

        # --------------------------------------------
        # SPAWN
        # --------------------------------------------
        @On(self.bot, "spawn")
        def on_spawn(this):
            self.bot.chat("/tp 1 -61 64") # beginning of the map
            # Start our repeating logic
            self.start_repeating_logic()

        # --------------------------------------------
        # END / DISCONNECT
        # --------------------------------------------
        @On(self.bot, "end")
        def on_end(this, reason):
            self.log(chalk.red(f"Disconnected: {reason}"))
            # Turn off events
            off(self.bot, "login", on_login)
            off(self.bot, "spawn", on_spawn)
            # off(self.bot, "kicked", on_kicked)
            # off(self.bot, "messagestr", on_messagestr)
            off(self.bot, "end", on_end)
            # Reconnect?
            if self.reconnect:
                self.log(chalk.cyanBright("Attempting to reconnect..."))
                self.start_bot()


    # ============================================
    # REPEATING TASK
    # ============================================
    def start_repeating_logic(self):
        """
        Called on spawn. Repeats every 0.2 seconds:
          1) Gathers snapshot (with blocks in 13*10*13 region)
          2) Calls action.py
          3) Applies the returned action
        """
        @AsyncTask(start=True)
        def periodic_logic(task):
            while True:
                try:
                    self.gather_bot_data()
                    # print("got data")
                    action = retrieve_action(self.view, self.bot.entity.position, self.bot.entity.yaw, self.pitch, self.velocity)
                    # print("got action")
                    if action:
                        # print("got action" + str(action))
                        self.apply_action(action)
                except Exception as e:
                    self.log(f"Error in repeating logic: {e}")
                time.sleep(0.2)



    def gather_bot_data(self):
        """
        Gathers the bot's position plus all blocks in a 15×7×15 region
        (±7 in X/Z, ±3 in Y) around the bot's integer location.
        Returns a dict with a 'blocks' key, so action.py can iterate over them.
        """
        pos = self.bot.entity.position
        center_x = int(pos.x)
        center_y = int(pos.y)
        center_z = int(pos.z)

        x_radius = 7
        y_radiustop = 3
        y_radiusbottom = 10  #for slime blocks
        z_radius = 7

        self.view = []
        for dx in range(-x_radius, x_radius + 1):
            for dy in range(-y_radiusbottom, y_radiustop + 1):
                for dz in range(-z_radius, z_radius + 1):
                    try:
                        self.view.append(world[(f"{center_x + dx}/{center_y + dy}/{center_z + dz}")])
                    except:
                         self.view.append("0")   # means block is not loaded



    def apply_action(self, action):
        """
        Clear all control states, then apply what's in 'action'.
        Example: ["jump", "forward", "sprint"]
        """
        self.bot.clearControlStates()
        for i in action:
            if i == "jump":
                self.bot.setControlState("jump", True)
            if i == ("forward"):
                self.bot.setControlState("forward", True)
            if i == ("back"):
                self.bot.setControlState("back", True)
            if i == ("left"):
                self.bot.setControlState("left", True)
            if i == ("right"):
                self.bot.setControlState("right", True)
            if i == ("sprint"):
                self.bot.setControlState("sprint", True)
            if i == ("sneak"):
                self.bot.setControlState("sneak", True)

# ============================================
# RUN
# ============================================
if __name__ == "__main__":

    # Total 7 bots + me
    bot1 = MCBot(f"ai_bot_1")
    bot2 = MCBot(f"ai_bot_2")
    bot3 = MCBot(f"ai_bot_3")
    bot4 = MCBot(f"ai_bot_4")
    bot5 = MCBot(f"ai_bot_5")
    bot6 = MCBot(f"ai_bot_6")
    bot7 = MCBot(f"ai_bot_7")



