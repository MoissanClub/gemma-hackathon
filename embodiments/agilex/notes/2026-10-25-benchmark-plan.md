# 2026-10-25: benchmark plan for the perception and action roles (two Pipers + G1)

The hackathon is over. We are extending the harness to general zero-shot tasks on two Piper arms and a Unitree G1, using larger
and cloud models where they help. Proposed split: a cloud model plans and points; local code does tracking, fast checks, grasping,
motion and safety. A cloud model may decide *what* to do and *where*. Local code always decides *how* to move and whether it is
safe, and a dropped connection leads to "hold and ask", never to motion. This note plans how to benchmark candidate models for the
four perception and action roles before relying on any of them.

**Nothing below is measured.** No number in this note exists yet; every expected outcome is a prediction until the benchmarks run.
The only measured figures quoted are existing E2B baselines, with their source.

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

## Next steps

1. Record and label the pointing dataset on the Pipers.
2. Build a common model wrapper and a JSONL format in `bench/`.
3. Run the cloud pointing models in parallel, then the local models one at a time.
4. Write up the results with n and per-class counts, and pick a model for each role.
