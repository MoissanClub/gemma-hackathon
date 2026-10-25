# 2026-10-25: benchmark plan for the perception and action roles (two Pipers + G1)

The hackathon is over. We are extending the harness to general zero-shot tasks on two Piper arms and a Unitree G1, using larger
and cloud models where they help. Proposed split: a cloud model plans and points; local code does tracking, fast checks, grasping,
motion and safety. A cloud model may decide *what* to do and *where*. Local code always decides *how* to move and whether it is
safe, and a dropped connection leads to "hold and ask", never to motion. This note plans how to benchmark candidate models for the
four perception and action roles before relying on any of them.

**Nothing below is measured**, except the section "Results so far" at the end. Every expected outcome is a prediction until the
benchmarks run. The other measured figures quoted are existing E2B baselines, with their source.

## What to measure per role

| Role | Candidates | Dataset | Metrics | Agents in parallel? |
| --- | --- | --- | --- | --- |
| Point at grasp spot / handle / keypoints | Gemini Robotics-ER, Claude, GPT, Molmo; Gemma E2B/E4B/12B as baseline | 100-200 recorded frames from the Pipers and the G1, hand-labelled grasp points, handles, corners | pixel error; cm error after 3D projection with depth; hit rate within 1/2/3 cm; refusal when the object is absent; p50/p90 latency at our site | cloud: one agent per model; local: serial |
| Track objects, 2D -> 3D | open-vocabulary detectors (e.g. GroundingDINO, OWLv2) + SAM2, vs Gemma boxes | short recorded sequences with labelled masks and IDs across steps | mask IoU; ID switches; 3D centroid error (cm); FPS on the Orin and on a workstation | serial on each GPU; one agent per machine |
| Quick yes/no checks during motion | Gemma E2B (baseline: one-word gate 320 ms on the Orin, `LEARNINGS.md` §2), E4B, a small cloud model | labelled frames: object in gripper / not, person present / not, task step done / not | accuracy with n and per-class counts; false positives (the dangerous kind); p50/p90 latency | local: serial; cloud: parallel |
| Grasp choice + path | grasp generators (GraspNet-style; top-down baseline from the mask's principal axis); planners (cuRobo, MoveIt) | recorded depth scenes + URDFs, scored in simulation or by geometry checks | rate of collision-free, reachable grasps; planning time; path length; joint-limit margin | yes: CPU / simulation work parallelizes |

Known from the G1 (`LEARNINGS.md` §1): E2B hallucinates a position with no object in view and a yes/no gate alone does not fix it,
so the pointing and check datasets need object-absent frames, scored as their own class.

## How agents split the work

1. **Dataset agent**: builds labelled frame sets from recorded runs. Images stay out of git (`.gitignore` covers `bench/images*/`
   and `runs/`).
2. **One runner agent per candidate**: wraps one model behind a common request/response interface and writes JSONL rows with timing
   and flags, as `bench/run_bench.py` does.
3. **Scheduler**: queues local GPU runs one at a time (one inference job at a time, `CLAUDE.md`) and never times repeated identical
   images (llama.cpp reuses the image KV; bench with `-np 1 --cache-ram 0`).
4. **Scorer agent**: computes metrics with n and per-class counts, keeping Mac, Orin and cloud results separate.
5. **Verifier agent**: spot-checks a sample of each model's outputs by hand rather than trusting the runners' own grading.

## Rules

- Benchmarks are offline. No benchmark moves a robot.
- Every accuracy with n and per-class counts. Mac numbers are not Orin numbers. Cloud latency is measured at our site, since it
  varies.
- Note what was downloaded (models, weights) in the run notes.

## Expected outcomes (predictions, not results)

- **Pointing**: cloud embodied-reasoning models likely beat Gemma by a wide margin, at ~1-3 s per call.
- **Tracking**: detector + SAM likely more precise than Gemma boxes, but heavier on the Orin.
- **Yes/no checks**: E2B probably good enough and fastest (E4B / 12B were no more accurate at perception on the G1,
  `docs/2026-10-22-orin-model-sizes.md`); the cloud only where E2B's accuracy is weak.
- **Grasp + path**: the top-down baseline may suffice for most rigid objects; a learned grasp generator matters for awkward shapes.

## Cost and effort

- Cloud: roughly 1-2k API calls per model for 200 frames, retries and variants included.
- Local: hours of serial GPU time across model sizes and machines.
- Labelling the frames is the real bottleneck, not compute.

## Access without paid APIs

- **Gemini Robotics-ER**: free tier via a Google AI Studio key (rate-limited: run slowly, overnight). The pointing model that matters
  most. Google lists ER 1.5 Preview; third-party listings show ER 1.6 (check AI Studio for the current one). Google reports pointing
  61% (1.5) -> 80% (1.6): their number, not ours.
- **Open pointing models** (Molmo, Qwen-VL, RoboBrain, RoboRefer): free locally, or a few dollars on a rented GPU. Same point-output
  format, one wrapper.
- **Claude / GPT**: no API key, so out of the first round. At most ~20 frames pasted by hand into the chat apps as a sanity check
  (no latency, not a benchmark). If one looks promising, pay-as-you-go (e.g. OpenRouter) is a few dollars for 200 frames.

## Verifying answers: fixed ground truth, never a model judge

- Public pointing sets ship masks: a point inside the mask is a hit, scored automatically.
- Own frames: click the point or draw the region once (a small labelling page, labels as JSON beside the frames, out of git);
  pixel error, then cm via D405 depth.
- Yes/no checks: hand labels; "in gripper" also from the gripper width against the measured diameter (`holding`).
- Spot-check ~20 answers per model by hand for what the scorer misses (malformed answers that happen to hit).

## Public datasets

| Role | Dataset | Use |
| --- | --- | --- |
| Pointing | Where2Place (`huggingface.co/datasets/wentao-yuan/where2place`) | 100 real cluttered scenes, sentence + free-space mask; first run |
| Pointing | RefSpatial-Bench (`huggingface.co/datasets/BAAI/RefSpatial-Bench`) | real scenes, precise masks; location / placement / unseen; mostly multi-step |
| Pointing | PointArena / Point-Bench (arXiv 2505.09990) | ~1,000 pointing tasks, 5 categories; blind pairwise arena |
| Pointing | RoboPoint data (arXiv 2406.10721) | grasp / place affordance points |
| Tracking | DAVIS, MOSE, YCB-Video | video object masks; YCB-Video has tabletop objects with 3D poses |
| Grasp | GraspNet-1Billion, Jacquard | depth scenes labelled with grasps |

Public sets check the pipeline and compare against published scores; the decision comes from our own Piper frames (none of these
has the wrist D405 view or our cans).

## Next steps

1. Run Where2Place and RefSpatial-Bench on Gemini Robotics-ER (free tier), Molmo and Gemma E2B: mask-scored, no labelling.
2. Record and label the pointing dataset on the Pipers.
3. Build a common model wrapper and a JSONL format in `bench/`.
4. Run the cloud pointing models in parallel, then the local models one at a time.
5. Write up the results with n and per-class counts, and pick a model for each role.

## Results so far

### Where2Place, Gemma 4 E2B (2026-10-25, laptop RTX 3070 via llama.cpp Vulkan; not Orin numbers)

`bench/pointing.py --model gemma`; the dataset's own question per image; score = share of the answer's points inside the
free-space mask, averaged over questions (RoboPoint's metric). Downloaded: Where2Place images, masks and questions (5.2 MB) into
`bench/images-where2place/` (git-ignored). Run log: `runs/bench/pointing-20261025T114928-gemma.jsonl` (git-ignored).

| Set | n | Mean score | Questions with any hit |
| --- | --- | --- | --- |
| all | 100 | 0.169 | 19 |
| seen | 70 | 0.182 | 14 |
| unseen | 30 | 0.137 | 5 |

- Latency p50 0.72 s, p90 1.00 s; ~223 prompt tokens.
- Format: 81 answers ignored the requested normalized `(x, y)` list and used Gemma's own `{"point": [y, x]}` on 0-1000 (mostly
  one point; mean 0.185); 19 used the requested list (mean 0.097). `[y, x]` was checked by drawing images 00 and 02: read that way
  the points land on the named mugs and in the drawer; read as `[x, y]` they land on unrelated objects.
- Hand spot-check of 20 random answers drawn on the images: scores match the pictures; no malformed answer scored a hit. E2B
  points at the object the sentence names ("left of the tissue box" -> on the tissue box), not at the free space the relation
  describes. It grounds the noun and drops the spatial relation (consistent with §1 of `LEARNINGS.md`: it perceives, it does not
  compose).

