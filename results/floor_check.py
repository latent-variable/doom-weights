"""Fable assumed the health_gathering floor is Doom-palette green with R:G:B about 0.47:1:0.40 at every
brightness, so R - 0.66 G + 0.5 B is about 0 on floor. Check it on real frames: the bottom quarter of the
screen, where the floor is, over a random walk."""
import numpy as np
from PIL import Image
from harness import env as E
g = E.Doom("health_gathering", "pixels", 5)
rng = np.random.default_rng(0); ratios = []; test = []
g.reset()
for t in range(400):
    x, r, done = g.step(int(rng.integers(5)))
    if done: g.reset(); continue
    s = g.last_screen.astype(float)
    if t == 40: Image.fromarray(g.last_screen).save("results/frames/health_gathering_real.png")
    floor = s[200:240].reshape(-1, 3)
    floor = floor[floor[:, 1] > floor[:, 0] + 10]          # green-dominant pixels only
    if len(floor) > 500:
        ratios.append(floor.mean(0) / floor.mean(0)[1])
        test.append(np.mean((floor[:, 0] - 0.66 * floor[:, 1] + 0.5 * floor[:, 2]) / 255))
r = np.array(ratios)
print("frames", len(r), "R:G:B median", np.median(r, 0).round(2), "spread (p10-p90) R", np.percentile(r[:, 0], [10, 90]).round(2), "B", np.percentile(r[:, 2], [10, 90]).round(2))
print("Fable's floor test R-0.66G+0.5B on floor: median", round(float(np.median(test)), 3), "(0 = the floor reads as floor)")
g.close()
g = E.Doom("basic", "pixels", 5); g.reset(); g.step(2)
Image.fromarray(g.last_screen).save("results/frames/basic_real.png"); g.close()
