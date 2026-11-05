# Write a neural network that plays Doom, by hand

You are going to write the weights of a small neural network so that it plays one Doom scenario well.
You will not train it. There is no data, no loss, no gradient, no optimizer. You design what each layer
should compute and set the numbers yourself, the way a programmer writes a program: the weights are the
program. Every weight starts at zero.

## The scenario: `deadly_corridor`

A long straight corridor. The player starts at one end facing down it; a green armour vest lies at the far end. Six monsters that shoot back (zombie soldiers with pistols, shotguns and chainguns) stand in alcoves along the corridor, three on each side. Played on the hardest skill level, so running straight for the vest usually gets the player killed. Reward: the change in distance to the vest each tic (+ for getting closer, - for moving away), +1,000 for reaching the vest, -100 on death. The episode ends on reaching the vest, at death, or after 2,100 tics. Higher score is better.

The game is real Doom (the ViZDoom research platform, a modified ZDoom engine, with Freedoom graphics), run headless.
Each step, the harness turns the screen into an input vector, your network scores each action, the
highest-scoring action is held for 4 game tics (35 tics are one second), and the game's own reward is
added to the score.

Actions, by index (the network's output order):
- 0: MOVE_FORWARD
- 1: MOVE_LEFT
- 2: MOVE_RIGHT
- 3: TURN_LEFT
- 4: TURN_RIGHT
- 5: ATTACK
- 6: MOVE_FORWARD + ATTACK

## What the network sees: `objects` mode

You see objects, not pixels. Per grid cell, channel 0 is the share of the cell's pixels that show a monster, channel 1 the share that show a pickup (a medkit, armour, ammo), and channel 2 the mean depth (distance to whatever is drawn there: 0 touching the camera, 1 far).

The input has 3601 numbers. The exact encoding is the function `encode` in `harness_src/env.py`, which is the
harness's real source code: read it, it is the ground truth. The grid is 30 rows by 40 columns of 8 x 8
pixel cells over a 320 x 240 render with a 90 degree horizontal field of view; index `(r * 40 + c) * 3 + k`
is row `r`, column `c`, channel `k`, and the game variables come after the grid.

## The network

Fixed shape (see `harness_src/net.py`, also the real source):

    h1 = relu(x  @ W1 + b1)    W1: (3601, 128)
    h2 = relu(h1 @ W2 + b2)    W2: (128, 128)
    q  =      h2 @ W3 + b3     W3: (128, 7)
    action = argmax(q), a tie going to the lowest index

The policy sees only the current input: it has no memory between steps.

## What to hand back, in this folder

1. `build.py`: a Python program that writes `weights.npz` with exactly the keys W1, b1, W2, b2, W3, b3,
   float32, the shapes above. It may import only `numpy` and the standard library, must run in a few
   seconds, and reads no files. The harness runs it again itself to get your weights.
2. `weights.npz`: the output of running `build.py`.
3. `explanation.md`: what each layer computes and why you expect it to score well, and your prediction of the
   network's average score on held-out games, with how confident you are.

## The rules

- **No learning of any kind.** No gradient descent, no backpropagation, no optimizers, no loss fitting, no
  least squares or regression fitted to data, no evolutionary or random search over weights. Every weight
  is a value you chose by reasoning about what it should compute.
- **No game.** Doom is not installed here and you must not install it or anything else, fetch anything from
  the internet, or look for game files. Python with numpy is all you have.
- You may run Python to build your weights and to test the network on inputs you construct yourself
  (a synthetic frame with a monster in column 12, say), to check it computes what you meant.
- **Practice: 10 rounds.** `./practice` plays your current `weights.npz` for 5 practice games (not the held-out ones) and writes the scores, the actions taken, and a few screens with the exact inputs your network saw into `practice_runs/<round>/`. You may look at all of it and adjust your weights by reasoning, but never fit weights to it (the first rule).
- Work only in this folder.

You are scored on the network's average game score over many held-out games of this scenario, which you
never see. Work until you are satisfied with it, then stop.
