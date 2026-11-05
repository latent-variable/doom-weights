"""The game the network plays, and exactly what it sees.

This file is given to the builder as source code. Everything a network's input
means is defined here and nowhere else.

A step: the harness encodes the current screen as a vector `x` (see `encode`),
the network maps `x` to one score per action, the highest score's action is held
for FRAME_SKIP game tics, and the game's own reward for those tics is added to
the episode's score.
"""
import os

import numpy as np

RENDER_W, RENDER_H = 320, 240   # the game renders at this size
GRID_W, GRID_H = 40, 30         # the network sees it averaged down to this grid
CELL = RENDER_W // GRID_W       # 8: each grid cell is an 8 x 8 block of pixels
FRAME_SKIP = 4                  # each chosen action is held for 4 tics (35 tics = 1 second)

# Object names the game reports, by what they are to the player.
MONSTERS = {"Cacodemon", "Demon", "MarineChainsawVzd", "Zombieman", "ShotgunGuy",
            "ChaingunGuy", "DoomImp", "HellKnight", "BaronOfHell", "LostSoul"}
PICKUPS = {"Medikit", "Stimpack", "GreenArmor", "BlueArmor", "Clip", "ClipBox",
           "Shotgun", "Shell", "ShellBox", "HealthBonus", "ArmorBonus"}
# Everything else (the player, bullet puffs, blood, corpses) is ignored.

SCENARIOS = {
    "basic": {
        "cfg": "basic.cfg",
        "actions": [["MOVE_LEFT"], ["MOVE_RIGHT"], ["ATTACK"]],
        "vars": [("AMMO2", 50.0)],
        "rules": (
            "A rectangular room with gray walls. The player spawns centred on one long wall, "
            "facing the opposite wall; a red, round flying monster (a Cacodemon) spawns at a random "
            "position along the opposite wall and stays roughly there. The player can only strafe "
            "left or right (it cannot turn) and shoot. One hit kills the monster. "
            "Reward: +106 for the kill, -5 for each shot, -1 for every tic. The episode ends when "
            "the monster dies or after 300 tics. Starting ammo is 50. Higher score is better; the best "
            "possible is a kill as early as possible with one shot."
        ),
    },
    "defend_the_center": {
        "cfg": "defend_the_center.cfg",
        "actions": [["TURN_LEFT"], ["TURN_RIGHT"], ["ATTACK"]],
        "vars": [("AMMO2", 26.0), ("HEALTH", 100.0)],
        "rules": (
            "A large circular room with the player fixed at its exact centre (it can only turn and "
            "shoot, never move). Five melee-only monsters spawn along the wall and walk toward the "
            "player; each dies from one pistol shot and respawns after a delay. "
            "Reward: +1 per kill, -1 on death. Ammo is limited (26 bullets), so death is inevitable; "
            "the episode ends at death or after 2,100 tics. Higher score is better: kill as many "
            "monsters as possible without wasting bullets."
        ),
    },
    "health_gathering": {
        "cfg": "health_gathering.cfg",
        "actions": [["TURN_LEFT"], ["TURN_RIGHT"], ["MOVE_FORWARD"],
                    ["MOVE_FORWARD", "TURN_LEFT"], ["MOVE_FORWARD", "TURN_RIGHT"]],
        "vars": [("HEALTH", 100.0)],
        "rules": (
            "A square room with walls around it and a green acid floor that damages the player "
            "every so often. Medkits are spread on the floor and more appear over time; walking "
            "over one restores health. There are no monsters and no weapon use. "
            "Reward: +1 for every tic alive, -100 on death. The episode ends at death or after "
            "2,100 tics. Higher score is better: stay alive by collecting medkits."
        ),
    },
    "deadly_corridor": {
        "cfg": "deadly_corridor.cfg",
        "actions": [["MOVE_FORWARD"], ["MOVE_LEFT"], ["MOVE_RIGHT"], ["TURN_LEFT"], ["TURN_RIGHT"],
                    ["ATTACK"], ["MOVE_FORWARD", "ATTACK"]],
        "vars": [("HEALTH", 100.0)],
        "rules": (
            "A long straight corridor. The player starts at one end facing down it; a green armour vest "
            "lies at the far end. Six monsters that shoot back (zombie soldiers with pistols, shotguns and "
            "chainguns) stand in alcoves along the corridor, three on each side. Played on the hardest skill "
            "level, so running straight for the vest usually gets the player killed. "
            "Reward: the change in distance to the vest each tic (+ for getting closer, - for moving away), "
            "+1,000 for reaching the vest, -100 on death. The episode ends on reaching the vest, at death, "
            "or after 2,100 tics. Higher score is better."
        ),
    },
}

MODES = {
    "objects": "3 channels per cell: monster share, pickup share, mean depth",
    "pixels": "3 channels per cell: mean red, green, blue",
}


def input_size(scenario):
    return GRID_W * GRID_H * 3 + len(SCENARIOS[scenario]["vars"])


def encode(mode, screen, labels_buf, labels, depth, game_vars, scenario):
    """The network's input vector, float32, length `input_size(scenario)`.

    Layout: a 30 x 40 x 3 grid (row, column, channel), flattened in that order, so
    the value for row r, column c, channel k sits at index (r * 40 + c) * 3 + k.
    Row 0 is the top of the screen, column 0 the left. After the grid come the
    scenario's game variables, each divided by its scale (see SCENARIOS).

    pixels:  channel k = the cell's mean red, green or blue, from 0 to 1.
    objects: channel 0 = share of the cell's 64 pixels that belong to a monster (0 to 1),
             channel 1 = share that belong to a pickup (0 to 1),
             channel 2 = mean depth of the cell, from 0 (touching the camera) to 1 (far);
                         the raw depth buffer divided by 255.
    """
    if mode == "pixels":
        img = screen.astype(np.float32) / 255.0                     # H, W, 3
    elif mode == "objects":
        img = np.zeros((RENDER_H, RENDER_W, 3), np.float32)
        for lab in labels:
            if lab.object_name in MONSTERS:
                img[..., 0][labels_buf == lab.value] = 1.0
            elif lab.object_name in PICKUPS:
                img[..., 1][labels_buf == lab.value] = 1.0
        img[..., 2] = depth.astype(np.float32) / 255.0
    else:
        raise ValueError(f"unknown mode {mode}")
    grid = img.reshape(GRID_H, CELL, GRID_W, CELL, 3).mean(axis=(1, 3))
    scales = [s for _, s in SCENARIOS[scenario]["vars"]]
    tail = np.asarray(game_vars, np.float32) / np.asarray(scales, np.float32)
    return np.concatenate([grid.reshape(-1), tail]).astype(np.float32)


class Doom:
    """One game. `reset()` gives the first input, `step(action)` the next one."""

    def __init__(self, scenario, mode, seed):
        import vizdoom as vzd
        self.vzd, self.scenario, self.mode = vzd, scenario, mode
        spec = SCENARIOS[scenario]
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, spec["cfg"]))
        g.set_window_visible(False)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_depth_buffer_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_available_buttons([getattr(vzd.Button, b) for b in self._buttons()])
        g.set_available_game_variables([getattr(vzd.GameVariable, v) for v, _ in spec["vars"]]
                                       + [vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HEALTH])
        g.set_seed(int(seed))
        g.init()
        self.g, self.n_vars = g, len(spec["vars"])
        self.actions = [[1 if b in a else 0 for b in self._buttons()] for a in spec["actions"]]

    def _buttons(self):
        seen = []
        for a in SCENARIOS[self.scenario]["actions"]:
            seen += [b for b in a if b not in seen]
        return seen

    def _obs(self):
        s = self.g.get_state()
        self.last_screen = s.screen_buffer
        return encode(self.mode, s.screen_buffer, s.labels_buffer, s.labels, s.depth_buffer,
                      s.game_variables[: self.n_vars], self.scenario)

    def reset(self):
        self.g.new_episode()
        return self._obs()

    def step(self, action, on_tic=None):
        """Hold `action` for FRAME_SKIP tics. Returns (next input or None, reward, done)."""
        self.g.set_action(self.actions[action])
        reward = 0.0
        for _ in range(FRAME_SKIP):
            self.g.advance_action(1)
            reward += self.g.get_last_reward()
            if self.g.is_episode_finished():
                break
            if on_tic:
                on_tic(self.g.get_state().screen_buffer)
        done = self.g.is_episode_finished()
        return (None if done else self._obs()), reward, done

    def stats(self):
        v = self.vzd.GameVariable
        return {"kills": self.g.get_game_variable(v.KILLCOUNT),
                "health": self.g.get_game_variable(v.HEALTH),
                "tics": self.g.get_episode_time()}

    def close(self):
        self.g.close()
