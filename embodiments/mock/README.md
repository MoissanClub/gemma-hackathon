# mock: the reference embodiment

No hardware, stdlib only. Use it to test `core/` and as the template for a new robot (`adapter.py`, under 80 lines).

- `media/scene_two_cans.jpg`: the G1's D435i head frame from 2026-10-19 (purple La Croix left, Chi sparkling lemon middle).
- `media/voice_order.wav`: "the purple one" in English; `media/voice_order_lang2.wav`: 白色的 ("white") in Chinese. Both from the
  G1's C922 mic.
- `locate()` fakes a point 40 cm ahead with y from the box centre and refuses boxes cut by the image border (the G1's rule);
  `pick()` always succeeds unless `--opt miss=true`; the handover "take it" clip is synthesized with macOS `say`.

```bash
python3 -B run.py -e mock --fake-gemma            # no server
python3 -B run.py -e mock --orders 2              # real Gemma, both clips
python3 -B run.py -e mock --typed                 # type orders
```

Seen with real Gemma on the Mac (2026-10-24): descriptions "purple sprite" / "silver chah lemon"; the English clip picks the purple
can; the Chinese "white" clip picks the purple can too (wrong: "white" is not in "silver chah lemon"; see LEARNINGS.md section 1).
