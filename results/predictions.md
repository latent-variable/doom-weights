# Each builder's own prediction, beside its score

The prediction is quoted from the run's explanation.md (its Prediction or Expected score section); the score is eval.json's mean.

## basic-objects-b0-20261104-211001
- model: claude-opus-5-5, practice rounds: 0, score: 77.8
- prediction: In an episode the player strafes until the monster is centred (from 0 to roughly 40 tics depending on where it spawned), then fires one shot, which takes a few tics to fire. That scores about 106 − 5 − (tics elapsed). Some risks: a miss caused by momentum drift during the fire delay, which costs −5 plus some tics, and later shots that may get pistol spread; or a little left/right oscillation if the strafe step oversh

## basic-objects-b0-20261104-211209-fable
- model: claude-fable-5-1, practice rounds: 0, score: 80.1
- prediction: **Average score on held-out games: about 79.** I would put roughly 70% on the range 74 to 84, and about 10% on something systematic being wrong (below 60). Nothing here was run against Doom. The number comes from a stand-in simulator I wrote from memory of the engine's rules (`sim_check.py`), in which this network averages 80.4 and kills with its first shot in every episode. The stand-in shares my assumptions, so it 

## basic-objects-b10-20261104-211208-opus
- model: claude-opus-5-5, practice rounds: 10, score: 78.0
- prediction: Score = 106 - 5 - tics. A monster spawned near the centre gives about 95. One at the far edge needs about 8 to 10 strafe steps, which gives about 60 to 65. Averaged over random spawn positions, I predict a **held-out mean of about 76, likely between 72 and 80**. I am fairly confident (about 80%) that it will be in that range. The main risk is a rare miss, which costs about 15 to 20 points for that game, or a spawn th

## basic-objects-b2-20261104-210553-smoke
- model: haiku, practice rounds: 2, score: 77.5
- prediction: Expected average on held-out games: about **70** (plausible range 55 to 85). Confidence: **moderate** on the mechanism, **low** on the exact number, since it rests on 5 games. The main costs are time and strafing granularity. The score is roughly `106 - 5 * shots - tics`. Each strafe step lasts 4 tics and moved the monster about 3 columns in the practice inputs, so the player often has to overshoot or re-align before

## basic-pixels-b0-20261104-211007
- model: claude-opus-5-5, practice rounds: 0, score: -265.3
- prediction: When it works: about 15–45 tics of strafing, depending on where the monster spawns, plus about 10 tics of shooting, and one shot (−5), sometimes two. That would give each game about 106 − 5 − 25 ≈ 70–90. The main risks come from assumptions I could not check without seeing the game: - (a) The Freedoom Cacodemon might be darker or less red than I assume, or the lighting very dim, so detection is weak. This is partly c

## basic-pixels-b0-20261104-211215-fable
- model: claude-fable-5-1, practice rounds: 0, score: -273.2
- prediction: **About 74 average on held-out games.** If my picture of the scene is right, I expect 72-79: about 18 tics of strafing on average, about 5 tics for the shot, one bullet, so 106 - 5 - 23 or so. Confidence is moderate: - **About 70%** that the average lands between 65 and 82. - **About 15-20%** that something I could not see breaks the detector and the average is far lower, possibly negative. The candidates are: - the 

## basic-pixels-b1-20261105-000137-opus
- model: claude-opus-5-5, practice rounds: 1, score: 76.6
- prediction: Average held-out score: **about 75** (my 80% interval is 55 to 88). I'm moderately confident in the perception part, which is checked against the real screens. I'm less sure about the physics: strafe speed, shot timing and the exact hit width are estimates. There is a small chance (~10%) of a systematic miss problem, which would pull the average down toward 30–50.

## basic-pixels-b10-20261104-211214-opus
- model: claude-opus-5-5, practice rounds: 10, score: 78.1
- prediction: Average held-out score of about **75** (likely range 70–80). The best possible is about 85–90 given the strafe time, and spawns near the centre score around 91. Confidence: fairly high (about 80%) that the mean is between 65 and 85. The main risks: - an occasional miss (roughly −20 each); - a monster spawning so close to a side wall that it is partly off-screen or blends into the corner seams. There is no sign of thi

## basic-pixels-b3-20261104-214006-opus
- model: claude-opus-5-5, practice rounds: 3, score: 77.5
- prediction: Held-out average: **about 77** (my 80% range is 70–83). I'm fairly confident about the basic behaviour: the colour feature separated cleanly in every frame I looked at, and all 10 games in rounds 2–3 were one-shot kills. Most of the uncertainty is in spawn positions, since monsters spawned near the side walls take longer to reach (around 60–65 points). There is also a small risk of an occasional miss when the monster

## deadly_corridor-objects-b0-20261104-212941-opus
- model: claude-opus-5-5, practice rounds: 0, score: -4.4
- prediction: **Predicted average score: about +250, with low confidence.** I would put an 80 % interval at roughly -50 to +800. The key uncertainties: - **The status bar.** If there is no status bar (horizon at y = 120), the upper-rows rule only sees monsters closer than about 150 units. The policy then degrades to "walk forward, shooting at whatever is centred", which probably scores about +100 to +300. - **The real map and timi

## deadly_corridor-objects-b10-20261104-214435-opus
- model: claude-opus-5-5, practice rounds: 10, score: 1340.6
- prediction: The expected held-out average is about **1,100**. I'd put a 70% interval at roughly **750–1,450**. - The score is almost entirely the probability of reaching the vest (about 2,280 vs about 300). - 10 practice games put that probability at about 0.4, with wide uncertainty (roughly 0.2–0.6). Known weaknesses: - Aim is coarse because of the 4-tic turn quantum. - On the hardest skill the pistol often loses the race again

## defend_the_center-objects-b0-20261104-211003
- model: claude-opus-5-5, practice rounds: 0, score: 16.4
- prediction: The ammo limit is the main cap. 26 bullets at about 1.3–2 bullets per kill gives roughly 13–20 kills at most. Turning time, monsters biting from behind, and the −1 for death take that down. **I predict an average score of about 8**, with a plausible range of 4–12. My confidence is low to moderate (about 50% that the true mean is within ±3 of 8). The biggest unknowns are: - how the labels buffer treats dying monsters;

## defend_the_center-objects-b0-20261104-211211-fable
- model: claude-fable-5-1, practice rounds: 0, score: 19.4
- prediction: **Average score on held-out games: about 13** (roughly 14 kills, minus 1 for the death that ends almost every game). Confidence is low. I have never seen the game run, and the score depends mostly on things I had to recall or guess (room size, monster speeds and damage, sprite sizes). My 50% interval is 8 to 18 and my 90% interval is 4 to 23. The hard ceiling is 25–26 (26 bullets). What I am fairly confident of: the 

## defend_the_center-objects-b0-20261104-215649-opus-rep2
- model: claude-opus-5-5, practice rounds: 0, score: 6.8
- prediction: Average score on held-out games: **about 6** (roughly 7 kills minus 1 for the inevitable death). I'd expect most games to land between 3 and 10. Confidence is moderate-to-low, about 50% that the mean falls in 4–9. The biggest uncertainty is whether the horizon really sits where I assumed. If it does not, expect about 1–3.

## defend_the_center-objects-b10-20261104-211210-opus
- model: claude-opus-5-5, practice rounds: 10, score: 23.6
- prediction: Average held-out score: **about 22–23** (range of plausible means 20–24.5). Most games should end with 24–25 (all ammo used, at most one or two misses). Occasionally the player gets bitten to death early by a monster that comes from behind while it is sweeping, and a game like that scores 10–20. This pulls the mean below the round-4 practice score of 24.2. I'm moderately confident (~70%) the mean will be ≥ 21.

## defend_the_center-pixels-b0-20261104-211009
- model: claude-opus-5-5, practice rounds: 0, score: 3.3
- prediction: Average score on held-out games: **about 4** (kills minus 1 for the inevitable death). My 50% interval is 1 to 8 and my 90% interval is −1 to 12. Confidence is low: the design rests on remembered facts about the scenario (status bar, sprite heights, colours, pistol timing) that I could not check here.

## defend_the_center-pixels-b10-20261104-211216-opus
- model: claude-opus-5-5, practice rounds: 10, score: 20.3
- prediction: Average held-out score **≈ 21** (15 practice games with the final design: mean 21.1, range 16–24). I'm about 80% confident the held-out mean lands between 19 and 23. Risks that could pull it lower: lighting or texture situations the practice screens didn't show, such as a large red tint while badly hurt, or blood gibs flying through rows 9–14 and drawing brief turns or shots.

## health_gathering-objects-b0-20261104-211005
- model: claude-opus-5-5, practice rounds: 0, score: 1991.0
- prediction: Score = tics survived, minus 100 if the bot dies. A full episode scores about 2,100. I expect most episodes to survive all or most of the 2,100 tics, with some deaths when medkits are sparse or the bot loses time at walls. - **Predicted average on held-out games:** about **1,400**. - **Range:** a plausible spread of 800–2,000, at about 60% confidence. - **Main uncertainty:** the depth-buffer scaling and the room's ac

## health_gathering-objects-b0-20261104-211213-fable
- model: claude-fable-5-1, practice rounds: 0, score: 2100.0
- prediction: **Average score on held-out games: about 1,850** (the maximum is 2,100, reached by surviving the whole episode). Confidence is moderate. I have never seen the game run, so the estimate is split three ways: | Case | My probability | Score I expect | |---|---|---| | The policy works as designed and medkit supply outpaces the acid | about 70% | 1,950 to 2,100 | | It works, but loses time (targets slipping out of view, s

## health_gathering-objects-b10-20261104-211212-opus
- model: claude-opus-5-5, practice rounds: 10, score: 2094.0
- prediction: The expected average on held-out games is **about 2050–2100**, and my point estimate is **~2090**. Every practice game ended at full health, so there is a lot of margin. The main risk is a rare game where the agent gets stuck against a wall or a corner with no medkit in view. I'm fairly confident (~80%) that the mean will be above 2000.

## health_gathering-pixels-b0-20261104-211011
- model: claude-opus-5-5, practice rounds: 0, score: 460.3
- prediction: For reference, an agent that does nothing dies at roughly 300 tics. A perfect one survives all 2,100 tics and scores about 2,100. If the floor is clearly green and medkits read as non-green, I expect most episodes to last a long time. I predict an **average of about 900 (plausible range 350-1800)**, and my confidence is low. The main risks are: - The floor's greenness is below the wall threshold (mean `G - (R+B)/2` <

## health_gathering-pixels-b0-20261104-211219-fable
- model: claude-fable-5-1, practice rounds: 0, score: 2100.0
- prediction: **Average score on held-out games: about 1500.** I hold this loosely: roughly a 50% chance the true average lands between 1200 and 2100, and the outcome is two-humped rather than bell-shaped. - If the screen looks the way I assumed (see "What I assumed"), I expect most episodes to run the full 2,100 tics, for an average near 1900–2100. - If one of the visual assumptions is wrong (for example a status bar is drawn, or

## health_gathering-pixels-b10-20261104-211218-opus
- model: claude-opus-5-5, practice rounds: 10, score: 2100.0
- prediction: Held-out average score: **about 2050** (most games reach the 2,100-tic limit). It could drop if some seeds leave the agent in a medkit-poor corner for a long time. I'm fairly confident (~75%) that the mean will be at least 1,900, and about 90% confident it will be at least 1,500.

