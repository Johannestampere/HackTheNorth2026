# Decisions and open questions

## Product decisions from the discussion

- A general physical AI companion selects useful tools and projects onto the environment.
- The sensor is an RGB camera. A model estimates depth from RGB inside perception; there is no depth camera. Audio comes later.
- The robot is beside the pool table, at a distance and approximately 1–2 m high, on a 360° rotating base. The earlier overhead setup is superseded.
- A general multimodal LLM inspects the initial image and chooses pool or generalization.
- Pool is implemented first. Generalization remains a separate future module.
- Three teammates own perception, planning, and projection respectively.
- Perception turns oblique/multiple captures into ball `(x,y,type)` data.
- Planning aims for eventual rack success. The first display produces cue alignment; later displays add predicted paths. A plan includes called ball/pocket and cue-stick speed from the start.
- Projection turns physical vectors into perspective-correct projector pixels.
- Initial planner: Pooltool 0.6.0, bounded direct-pot angle/speed search, sampled uncertainty utility, and bounded four-shot own-turn lookahead (4,000-call default budget). See [implementation](planning.md).

## Engineering defaults in this scaffold

These are concrete starting conventions, not claims about selected hardware or completed algorithms:

- Python 3.11+, a standard-library-only shared core, immutable dataclasses, structural protocols, and constructor-based dependency injection.
- Table-length units: long side = 1, short side = short/long ratio, top-left origin with x right and y down. Radii and all distances use the same unit; pixels stay at acquisition/rendering boundaries.
- Known/measured table geometry provided to perception instead of requiring automatic geometry estimation in v1.
- File-backed camera manifests and versioned JSON fixtures for independent development.
- RGB-only capture contract; estimated depth stays internal to perception.
- Explicit partial/unusable/no-shot outcomes instead of fake successful results.
- Observation/table/calibration/pose IDs to catch inconsistent handoffs.
- Schema 3 uses explicit WPA 8-ball context. MVP scope is after the break with assigned groups and a placed cue ball, including legal eight-ball finishes. A complete rules engine is not implemented.
- RGB8 byte frames with PPM export for dependency-free inspection.
- A fixed-pose homography target first; rotating target selection and scan orchestration remain future application work.
- No automatic model downloads, LLM calls, hardware commands, commits, or pushes.

## Still to decide with the team

| Question | Why it matters | Owner(s) |
| --- | --- | --- |
| Which RGB camera and depth-estimation model? | Image resolution, model accuracy/scale, inference latency | Perception |
| Are cameras and projector rigidly mounted to the same rotating assembly? | Extrinsics and pose calibration | Perception + projection |
| Does 1–2 m height mean above floor or cloth, and what is table distance? | Occlusion, effective resolution, projector throw | Whole team |
| Is pitch adjustable, and how repeatable is the base? | Whether the table can be reached and alignment retained | Hardware + projection |
| What are the table, ball, and pocket proportions? | Width/long-side ratio, normalized collision radii, calibration | Whole team |
| What is the measured physical table length? | Replace the explicit 2 m simulation assumption | Planning |
| Markers/manual calibration acceptable for the demo? | Fast reliable setup versus automatic localization scope | Perception + projection |
| Required localization/projection error and maximum latency? | Acceptance thresholds and feasible shots | Whole team |
| How should assumed errors/utility weights be calibrated, and how deep should future search go? | Real execution reliability and rack strategy | Planning |
| How does the operator supply the current group and confirm cue-ball placement? | Explicit game context for each planning request | Planning + integration |
| Can the projector cover the cue guidance at all relevant locations? | Target pose selection and multi-region presentation | Projection |
| Which teammate owns scanning/application integration? | Work that crosses the three stage boundaries | Whole team |
| Which LLM/provider and routing behavior on uncertainty? | Future tool router implementation | Integration |

Do not block fixture-based work on these answers. Do resolve physical setup and calibration questions before claiming hardware accuracy or a complete observable table state.
