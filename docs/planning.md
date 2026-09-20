# Pool planning: simulation and uncertainty scoring

The planner now runs real Pooltool 0.6.0 simulations. Its existing schema-3 input/output contract is unchanged. Perception and projection do not import Pooltool.

## Install and run

Use Python 3.11–3.13 (Pooltool 0.6 does not support 3.14). From the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[planning]'
PYTHONPATH=src .venv/bin/python -m companion.app plan \
  --state fixtures/table_states/direct_shot.json \
  --game fixtures/game_contexts/solids.json \
  --table-length-m 2.0 --output artifacts/shot_plan.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Pooltool's dependencies include Panda3D even though this code opens no 3D window. Its first import creates configuration files under `~/.config/pooltool`. The adapter explicitly constructs the release's default physics resolver so personal resolver settings do not affect results. Initial imports/JIT compilation can take longer than subsequent runs.

## Watch a game

```sh
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo
# A short two-shot endgame, using the existing fixture:
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo \
  --state fixtures/table_states/direct_shot.json --output artifacts/pool-endgame.html
```

Open `artifacts/pool-game.html` in a browser. It is a standalone offline Canvas replay with play/pause, restart, shot selection, timeline scrubbing, playback speed, and predicted paths. The adjacent JSON stores trajectories and shot summaries. Generated files are ignored by Git; teammates generate them with the same command.

The default seed is 7; change it with `--seed`. This is a generated 16-ball spread layout, not a simulated break. Groups are assigned; solids start. Each simulated successful called pot keeps the turn. The actual settled positions feed the next planning call. A legal called eight-ball pot ends the rack.

**Demo limitation:** if the planner finds no direct pot, the demo logs a player switch without executing a shot. That is a visualization convention, not a legal pass or implemented safety. If neither player has a supported shot, the demo stops without declaring a winner. It also stops at `--max-shots` (default 32). By default execution is nominal. Pass `--execution-seed 23` to sample aim/speed errors independently from the planner trials. Legal misses change turns; fouls stop for unimplemented ball-in-hand placement. Execution keeps exact observed positions (perception error is sampled during planning only). Neither mode establishes real human accuracy or win probability.

## File ownership and flow

- `service.py`: configuration, input checks, bounded search, scoring, and `ShotPlan` construction. `last_selection` exposes private diagnostics only for tests/demo.
- `candidates.py`: private candidate record, legal target group, and finite swept-ball path checks.
- `physics.py`: all Pooltool imports, coordinate conversion, table construction, ghost-ball candidates, simulation, outcome classification, and trajectories.
- `lookahead.py`: bounded recursive own-turn search from multiple sampled outcomes.
- `uncertainty.py`: repeatable error samples, empirical outcome counts, utility and diverse shortlisting.
- `demo.py` / `replay.html`: synthetic self-play runner and offline viewer.

`PooltoolPlanner.plan(state, game)` performs:

1. Reject unsupported game contexts, incomplete observations, missing cue/eight, unknown types, overlapping balls, and balls too close to rails for this initializer.
2. Construct stationary balls on a standard six-pocket table. Preserve observation ball IDs; map pocket IDs by location.
3. Generate legal ball/pocket candidates using Pooltool's ghost-ball and potting-angle helpers. Check swept cue/object paths, ghost-ball bounds, pocket jaws, and cuts no greater than 75 degrees.
4. Search up to 16 geometric candidates, five cue speeds (0.35, 0.5, 0.7, 0.95, 1.25 table lengths/s), and offsets of 0, -0.75, +0.75 degrees. Maximum 240 nominal simulations per call.
5. Inspect collision and pocket events. Discard scratches, illegal first contact, early/wrong-pocket eight-ball losses, and failed called pots. A shot must pocket a ball or hit a rail after first contact.
6. Build a diverse shortlist of up to 12 nominal successes, round-robin across ball/pocket pairs. Evaluate each against the same 24 perturbed layouts/strikes. Rank by empirical utility (below); break ties by success rate, fewer rack losses/fouls, nominal win, follow-up availability/ease, current geometric difficulty and speed. Then compare the top three using the multi-shot search described below.
7. Return a forward unit cue vector, cue-tip speed, called ball/pocket, actual simulated cue center at first contact, and event-to-event path guides. If none succeeds, return `NoFeasibleShot`; this does not mean no legal shot exists.

The adapter inspects Pooltool events directly for the supported rule subset, rather than initializing its full rules engine with a reconstructed game history. Breaks, ball-in-hand placement, jumped balls, physical cue fouls, and safeties remain unsupported. Expected input limitations return typed outcomes; unexpected simulator errors are not silently converted into a successful recommendation.

## Units and calibration

Shared geometry remains long side = 1, width = short/long, origin top-left. The adapter uses:

```text
engine_x_m = shared_y * physical_length_m
engine_y_m = shared_x * physical_length_m
engine_cue_speed_mps = shared_cue_speed * physical_length_m
shared_direction = (sin(engine_phi), cos(engine_phi))
```

Default physical length is an explicit **2.0 m assumption**, overrideable through `PlannerConfig` or `--table-length-m`. It is not inferred from RGB. Both ball radius and pocket mouth widths use this same scale. The MVP requires four corner pockets and two long-rail midpoint pockets; corner widths must match each other, as must side widths. Unsupported geometry is rejected, not averaged away.

Pocket jaw shape, pocket depth, mass, friction and restitution use uncalibrated Pooltool defaults; cushion height is set relative to ball radius. Cue offsets and elevation are explicitly zero. Even correct geometry does not make predicted travel distances accurate on a real table until these physical parameters and player speed are calibrated.

## Uncertainty scoring

Defaults are explicit uncalibrated assumptions, configurable through `PlannerConfig.errors` (`ErrorModel`):

- Aim: zero-mean Gaussian error, standard deviation 0.5 degrees.
- Speed: uniform multiplier between 0.9 and 1.1.
- Ball position: uniform disk of radius `Ball.position_uncertainty`, or 0.001 table lengths when absent. A reported zero means no position perturbation. Treating the contract's generic uncertainty as a radius bound is this planner's assumption; agree/calibrate its meaning with perception before real use.

Positions are independently proposed, then the whole layout is rejected if centers overlap or enter the rails. Sampling is therefore conditioned on valid layouts, not clipped. If 200 attempts cannot produce a valid layout, planning returns `InsufficientInformation`. Trials preserve intended aim/speed relative to the observed layout; they do not re-aim using hidden true coordinates. All shortlisted strikes see the same seeded trials (default seed 17).

Each trial contributes exactly one outcome, in this precedence:

| Outcome | Utility |
| --- | --- |
| Rack loss | -5 |
| Foul (excluding rack loss) | -2 |
| Legal rack win | +3 |
| Legal called pot | +1 |
| Legal miss | 0 |

The mean utility ranks candidates for the lookahead shortlist. These are engineering weights, not measured game values; they penalize dangerous shots without implementing opponent lookahead. Sampled success rate is `(wins + legal called pots) / trials`. `SelectedShot.stats` includes all five counts; it stays private and does not change `ShotPlan` or schema 3. Displayed paths remain the nominal prediction, not an uncertainty envelope. A 24-trial estimate is coarse (one trial changes a rate by about 4.2 percentage points), selection uses the same trials as reporting, and zero observed fouls is not proof of zero risk. No minimum success threshold or calibrated real-world probability is claimed.

First-shot cost: at most 240 nominal simulations + 12 × 24 uncertainty trials = 528 simulations. Seeds make runs repeatable; increasing `uncertainty_trials` trades latency for less sampling variation. Diverse shortlisting still prunes the search, so the winner is only best among evaluated candidates. A bounded four-shot own-turn search is now enabled by default; opponent replies are not searched.

```sh
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo \
  --execution-seed 23 --output artifacts/pool-game-noisy.html
```

The noisy replay keeps the nominal guides visible while balls follow the independently sampled strike. It reports trial success rates and records aim/speed error, empirical statistics, and executed outcomes in JSON. On seed 23 the tested run includes a legal miss and a later scratch, then honestly stops for placement.

## Three- and four-shot lookahead

`PlannerConfig.search_depth` counts the current shot and accepts 1–4 (default 4). `lookahead=False` or depth 1 selects the immediate-only baseline. The shared `ShotPlan` still describes just the next strike; every real observation triggers a fresh decision.

1. Keep the top three first strikes by uncertainty score and sample up to three evenly spaced first-shot outcomes for each.
2. Misses, fouls, losses, and completed wins stop the branch with no continuation bonus. They remain in the averaging denominator.
3. After a legal nonterminal pot, search the next observed position. Each node evaluates up to two ball/pocket candidates, at most 30 nominal strikes, then up to three successful strikes with up to five shared uncertainty samples (15 calls).
4. Keep the two strongest actions at that node. Expand up to two sampled outcomes of each to the next depth. This is a narrow, pruned own-turn tree, not exhaustive game search.
5. Apply the same scoring recursively: **immediate mean utility + 0.5 × average continuation value**. Future value is floored at zero because the model has no safety/opponent alternative. First-shot ranking breaks root ties.

Default hard cap: **4,000 physics simulations per decision**. First-shot search needs at most 528. The remaining quota is split evenly between root branches, then between child actions/outcomes. Each node reserves calls for uncertainty testing and respects its quota; unused sibling quota is not redistributed. A narrow layout or terminal branch can use much less than the configured cap. This limits simulation count, not wall-clock time; Pi timings have not been benchmarked.

If depth or budget is exhausted, the search returns the value already established at that node. It retains a legal nominal first-shot recommendation. `reached_depth` reports the deepest evaluated level, including alternative branches; it does not mean every branch reached that depth. `budget_limited` means some subtree received insufficient quota for its full local search, even if other branches left budget unused. `paths` lists one illustrative target-ball sequence per root outcome. These are conditional examples, not guaranteed executable instructions or win probabilities.

Deeper branches re-plan their aim after observing the previous settled layout. Exact observations at that boundary are optimistic; each subsequent strike still receives noisy position/aim/speed trials. First-shot errors use 24 trials, deeper errors at most five; this coarse sampling can miss risks and should not be presented as calibrated confidence. The branch stops if its settled state cannot be represented by our table contract.

```sh
# One decision on the generated spread rack, with replay diagnostics:
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo \
  --max-shots 1 --search-depth 4 --simulation-budget 4000 \
  --output artifacts/pool-depth-four.html

# The plan and project commands accept the same options:
PYTHONPATH=src .venv/bin/python -m companion.app plan \
  --state fixtures/table_states/direct_shot.json \
  --game fixtures/game_contexts/solids.json \
  --search-depth 3 --simulation-budget 2500 --output artifacts/shot-plan.json
```

The replay reports requested/reached depth, truncation, actual lookahead calls, and illustrative target sequences. Depth 2 with a 1,000-call budget remains available for comparison. Settings whose first-shot worst-case cost already exceeds the budget fail validation rather than silently overrunning it.

This can search a complete runout when the remaining own balls plus the eight fit within four shots. It cannot predict the whole game: opponent decisions, safeties, ball-in-hand, banks, and breaks remain unsupported.

## Next milestones

1. Calibrate rolling distance and cushion/pocket behavior on the physical table.
2. Add opponent replies, deeper lookahead, banks and a contract for safeties. A broader search is needed to keep games progressing without demo skips.
3. Consider learned value estimates only after collecting simulator/real outcome data.

Sources: [Pooltool release](https://pypi.org/project/pooltool-billiards/0.6.0/), [potting helper implementation](https://github.com/ekiefl/pooltool/blob/v0.6.0/pooltool/ai/pot/core.py), [table specification](https://pooltool.readthedocs.io/en/latest/resources/table_specs.html).

## Compare every attempted angle for one shot

```sh
PYTHONPATH=src .venv/bin/python -m companion.pool.planning.demo \
  --compare-angles --output artifacts/pool-angle-comparison.html
```

This uses the direct-shot fixture by default (`--state` overrides it). Every replay starts from the exact same positions. The dropdown includes all nominal ball/pocket candidates, each at all five speeds and three aim offsets, including failed pots and fouls. Labels show the absolute angle in the shared frame (0° right, 90° down), engine-frame offset, speed in table lengths/second, and outcome. `SELECTED` identifies the actual recommended strike. Play steps through attempts; the dropdown selects any attempt directly. This is the initial candidate grid, not all random uncertainty samples or future-shot searches. Re-simulation is deterministic with the same adapter/configuration.
