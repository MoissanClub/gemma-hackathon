# Learnings that carry across embodiments

Distilled from five weeks on the Unitree G1 (Sept 16 - Oct 24, 2026). Each line says where the evidence is. Mac numbers are
MacBook Pro M4 Pro; Orin numbers are the G1's Jetson Orin NX 16 GB at MAXN. The source kitchens of the fridge benchmark are generated
images: those results are a diagnostic, not navigation capability.

## 1. How to ask Gemma 4 E2B (holds for any robot)

- **One direct perception question per request, asked after the image; code does geometry and sequencing.** E2B without thinking
  perceives well (99% one-word, 97% tool argument, 100% box) but cannot compose percept -> plan: every "plan the actions" variant
  collapses to `[forward]` (23%). Thinking + `plan` tool was 63% at 2.1-2.6 s. Reasoning budgets under 1 s do not help.
  `docs/LATENCY_RESULTS.md`
- **A question in the system turn or before the image loses accuracy** (box detection excepted). `docs/LATENCY_RESULTS.md`
- **Box detection with an assistant prefill is the most robust format**: 78/78, unaffected by temperature, no left/right flips. It
  cannot say "nothing there" (the prefill forces a box). `face_target(position)` tool calls sometimes say `left` for a target on
  the right, never the reverse. `docs/LATENCY_RESULTS.md`
- **Presence: never trust a yes/no gate alone.** With no object in view E2B hallucinates a position; the yes/no gate and a 4-way
  answer are both partial fixes. On the G1, presence and colour were decided per detected crop ("what colour is this can", one
  word): 0 wrong picks, 2 correct refusals. `embodiments/g1/notes/2026-10-19-office-grab.md`
- **Choosing by description works from audio in any language**: Gemma hears the order natively (no speech recognizer) against a
  numbered list of crop descriptions and answers the number: 0.26-0.34 s on the Orin, any language. Two failure modes seen:
  - an order-free clip ("Hm.") was answered `1` (G1, 2026-10-23): always confirm out loud before acting;
  - **the guest's word must appear in the description.** A Chinese clip asking for 白色的 (white) is heard as "white", but against
    "purple sprite, on the left; silver chah lemon, on the right" it picks 1 (wrong); against "purple sprite; silver chi" it picks 2
    (right) (shared harness, Mac, 2026-10-24, n=1 each). Silver vs white is the gap: add a one-word colour per crop to the description
    (`Gemma.crop_colour`) before relying on colour orders.
- **"Take it" by transcript + keywords beats a yes/no judgement** (14/14 vs 9/14 on synthetic clips, Mac), but real let-go words were
  misheard on the G1 ("A cake.", "Packet"): keep a key or timeout as the fallback. The robot hears its own TTS: never record within
  4 s of a spoken sentence. `embodiments/g1/notes/ROBOT_RULES.md`
- **Third-person "is it in the hand" is right after the lift (8/9), not before.** Prefer the gripper's own sensing (finger stall,
  gripper width) when it has any. `embodiments/g1/notes/ROBOT_RULES.md`
- The native tool grammar does not enforce enums or ranges (validate client-side); the server rejects `grammar` with `tools`; the
  chat template sorts tool arguments alphabetically. `docs/LATENCY_RESULTS.md`

## 2. Latency levers (on-device)

- Thinking **off** (-2 s) -> 480x336 frame (70 image tokens, llama.cpp's floor; -135 ms) -> cached system prefix (-25 to -150 ms) ->
  few output tokens (~9 ms each on the Mac, ~28 ms on the Orin) -> `--swa-full -c 4096` (-35 to -55 ms). Prompt processing is ~90% of
  a short request. Server flags: `serve/run_llama.sh` (Mac), `serve/serve-orin.sh` (Orin).
- **Orin NX, E2B** (llama.cpp CUDA, MAXN, `jetson_clocks`): one-word gate 320 ms, `face_target` tool call 686 ms, box 772 ms, spoken
  order 0.22 s as two words / 0.73 s as a tool call; Mac-identical accuracy on 78 frames; 5.7 GB RAM.
  `embodiments/g1/notes/2026-10-18-orin-sunday.md`
- **Model size**: perception stays on E2B (E4B / 12B are no more accurate and 1.4-1.8x / 2.3-3.3x slower). Writing code for an
  unseen task is where size matters: E2B 2/9, **E4B 8/9 at 4.2 s**, 12B 8/9 at 12.2 s against the same small robot API
  (`core/robot_api.py`). `docs/2026-10-22-orin-model-sizes.md`
- MTP drafting is a net loss on the Mac. LiteRT-LM on the Orin: CPU 4-15x slower; GPU (0.18.0, WebGPU over Vulkan) hears the order
  with llama.cpp's accuracy at 0.56-0.63 s p50 (llama.cpp 0.17-0.22 s), and both engines fit together. `docs/2026-10-22-litert-gpu.md`
- **Never time repeated identical images**: llama.cpp reuses the image KV and reports fantasy latencies. Bench with
  `-np 1 --cache-ram 0`, rotate images (`bench/run_bench.py` flags `image_cache_hit`), one inference job at a time.

## 3. Physical skills (the G1's lessons, phrased for any arm)

- **Measure the setup at runtime; never hard-code a distance.** Table plane and edge from depth (level fit: free RANSAC leaned 3-16
  degrees through can sides), object axis = front-surface depth + half the classified diameter (a 3.3 cm error put a can at the
  fingertips). Every measured number lives in a calib file with provenance (`embodiments/g1/brain/calib.json`), and a CLI override
  is logged as a deviation. `embodiments/g1/notes/ROBOT_RULES.md`
- **Feed the sag forward, then correct on the measured end effector.** Uncorrected moves on the G1 sagged 2.5-4.6 cm at the high waypoints; v2 placed the
  hand beside the can within 2-4 mm on every run by feeding the sag forward and correcting until the measured palm is within 6 mm.
  A correction must move the measured point, not the commanded one. `embodiments/g1/notes/2026-10-23-v2-grasp.md`
- **Plan the whole path before moving; cap every per-step joint change; reach by feasibility search, not a hand-tuned lane.**
  `embodiments/g1/brain/README.md` (geom.plan_reach)
- **Contact is a tracking-error rise or the hand lagging the command**: stop there, never close on an abort. Approach from the side
  with the thumb opposed; a forward approach dragged the fingers over the table edge and pushed the can 19 cm.
  `embodiments/g1/notes/2026-10-19-office-grab.md`
- **The gripper's stall pattern is the grasp signal and says what went wrong** (G1: fingers 0.40-0.55 on a can, 0.90 empty; thumb 0.9
  with fingers stalled = object ahead of the palm; all 0.9 = pushed away). One mapped correction, then give up or ask for help.
- **A stop holds; it never retreats on its own.** The robot says its way home and runs it only on the operator's `y`. When the
  program exits away from home the controller keeps the last target (the G1 stayed stiff at the offer pose).
  `embodiments/g1/brain/README.md`, `embodiments/g1/notes/INCIDENTS.md`
- Narrow cameras cut objects at the border (the G1's 56 degree head view puts cans more than ~15-18 cm left at the edge): refuse a
  box touching the border and ask for the object to be moved, in words.

## 4. Process

- Design and code on the laptop, then test on the robot (`docs/DEV_WORKFLOW.md`): a plan the owner approves, desk checks, a mock run,
  perception replay, a robot session that only executes the run plan, a fresh post-mortem. No code edits on the robot's computer.
- Do not tune small deltas on the robot: reproduce the fault offline, re-derive from what is known, let vision feedback absorb the
  error. One battery was burned finding a 3.3 cm frame bug that an offline test would have caught (Oct 20).
- Report accuracy with n and per-class counts; seed every sampled request differently.
