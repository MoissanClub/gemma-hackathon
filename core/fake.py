"""Scripted Gemma answers in place of llama-server (`run.py --fake-gemma`): the loop runs with no model, for wiring tests only.
A stand-in for `core.client.timed_chat`; it reads the question in the payload and answers like the real model would."""
import re

DESCS = ["purple sprite", "silver chi lemon", "red cola", "green lime"]
_n = [0]                         # descriptions restart with every detection: one scene, the same labels


def chat(base_url, payload, **kw):
    msgs = payload["messages"]; user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    parts = user if isinstance(user, list) else [{"type": "text", "text": user}]
    text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text").lower()
    system = msgs[0]["content"].lower() if msgs[0]["role"] == "system" else ""
    if "detect every" in text:
        _n[0] = 0; ans = "120, 40, 600, 210], \"label\": \"can\"}, {\"box_2d\": [330, 390, 650, 510], \"label\": \"can\"}]"
    elif text.startswith("detect"): ans = "330, 390, 650, 510"
    elif "describe this" in text: ans = DESCS[_n[0] % len(DESCS)]; _n[0] += 1
    elif "which can does the guest want" in text:
        said = text.split("the guest says:")[-1] if "the guest says:" in text else ""
        cans = dict((m.group(2).strip(), m.group(1)) for m in re.finditer(r"(\d+): ([^;.]+)", system))
        words = {w for w in re.findall(r"[a-z]+", said) if len(w) > 3}
        ans = next((n for d, n in cans.items() if words & set(re.findall(r"[a-z]+", d))), "1" if not said else "none")
    elif "transcribe" in text: ans = "The purple one. Okay, take it, thank you."   # agrees with the scripted audio choice (1)
    elif "yes or no" in text: ans = "yes"
    else: ans = "ready"
    return {"content": ans, "timings": {}, "total_s": 0.001, "tool_calls": [], "reasoning": ""}
