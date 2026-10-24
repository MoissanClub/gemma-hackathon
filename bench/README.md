# bench: latency and accuracy of the Gemma request formats

Findings: `docs/LATENCY_RESULTS.md` (Mac) and `docs/2026-10-22-orin-model-sizes.md` (Orin: E2B / E4B / 12B, codegen probe).

**Images are not committed** (about 90 MB generated, 7 MB of source renders). To run the fridge benchmark, copy the three source
kitchens `assets/kitchen-source.png`, `assets/kitchen-left-source.png`, `assets/kitchen-right-source.png` from the `harness` repo
into `assets/` at this repo's root, then `python3 bench/make_dataset.py` (macOS `sips`) writes `bench/images/` to match the labels
in `dataset.json` (78 views) and `absent.json` (14 fridge-free views). On Linux, copy the generated `bench/images*/` folders instead.
`codegen_probe.py` needs no images.

| File | What |
| --- | --- |
| `run_bench.py` | serial sweep of `variants.py` over the dataset -> `experiments/<ts>-latency-<tag>/{requests.jsonl,summary.json}`; rotates images and flags `image_cache_hit` |
| `variants.py` | every prompt / tool / grammar / image-size variant (`run_bench.py --list`) |
| `report.py` | Markdown tables from summaries |
| `codegen_probe.py` | the model writes `def task(robot):` for an unseen task against a mock robot API (`--set basic|hard`); model code runs where the script runs (run on the laptop, tunnel to the robot's server) |
| `serve.sh` | restart the port-8080 server with flags (its launch line goes into each run's manifest) |
| `flag_sweep.sh`, `batching.py`, `text_mapping_probe.py`, `vit_split.py` | server-flag A/B, N actions per call, text-only percept -> plan control, vision-encoder split from a `-lv 4` log |
| `run.py`, `tools.py`, `prompts/`, `*-defaults.json`, `evaluation-sampling.json`, `cases.json` | the original one-response harness the benchmark imports (`assess`, tool schemas) |
| `client.py` | link to `core/client.py` |

```bash
bench/serve.sh dev -np 1 --cache-ram 0 -c 4096 --swa-full
python3 -B bench/run_bench.py --list
python3 -B bench/run_bench.py --tag dev --variants "face-tool-480x336,where-free-off-480x336,box-prefill-480x336"
python3 -B bench/codegen_probe.py --set basic --tag e2b
```

Serial and GPU-exclusive: never two inference jobs at once. Never time repeated identical images.
