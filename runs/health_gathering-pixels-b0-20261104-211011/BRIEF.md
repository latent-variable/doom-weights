# Write a neural network that plays Doom, by hand

You are going to write the weights of a small neural network so that it plays one Doom scenario well.
You will not train it. There is no data, no loss, no gradient, no optimizer. You design what each layer
should compute and set the numbers yourself, the way a programmer writes a program: the weights are the
program. Every weight starts at zero.

## The scenario: `health_gathering`

A square room with walls around it and a green acid floor that damages the player every so often. Medkits are spread on the floor and more appear over time; walking over one restores health. There are no monsters and no weapon use. Reward: +1 for every tic alive, -100 on death. The episode ends at death or after 2,100 tics. Higher score is better: stay alive by collecting medkits.

The game is real Doom (the ViZDoom research platform, a modified ZDoom engine, with Freedoom graphics), run headless.
Each step, the harness turns the screen into an input vector, your network scores each action, the
highest-scoring action is held for 4 game tics (35 tics are one second), and the game's own reward is
added to the score.

Actions, by index (the network's output order):
- 0: TURN_LEFT
- 1: TURN_RIGHT
- 2: MOVE_FORWARD
- 3: MOVE_FORWARD + TURN_LEFT
- 4: MOVE_FORWARD + TURN_RIGHT

## What the network sees: `pixels` mode

You see the screen itself: per grid cell, the mean red, green and blue of its 64 pixels, each from 0 to 1. Nothing tells you what is a monster or a wall; that is in the colours.

The input has 3601 numbers. The exact encoding is the function `encode` in `harness_src/env.py`, which is the
harness's real source code: read it, it is the ground truth. The grid is 30 rows by 40 columns of 8 x 8
pixel cells over a 320 x 240 render with a 90 degree horizontal field of view; index `(r * 40 + c) * 3 + k`
is row `r`, column `c`, channel `k`, and the game variables come after the grid.

## The network

Fixed shape (see `harness_src/net.py`, also the real source):

    h1 = relu(x  @ W1 + b1)    W1: (3601, 128)
    h2 = relu(h1 @ W2 + b2)    W2: (128, 128)
    q  =      h2 @ W3 + b3     W3: (128, 5)
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
- **No practice.** Your network never plays before it is scored. You have never seen this game run and will not.
- Work only in this folder.

You are scored on the network's average game score over many held-out games of this scenario, which you
never see. Work until you are satisfied with it, then stop.
