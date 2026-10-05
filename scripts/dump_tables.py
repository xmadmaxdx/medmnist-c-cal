"""Dump corrupted per-severity tables from full JSONs for paper transcription."""
import json

for f in ("outputs/outputs_json/blood-full.json", "outputs/outputs_json/derma-full.json"):
    d = json.load(open(f, encoding="utf-8"))
    print("=" * 20, f)
    print("temp", round(d["temperature"], 4))
    for m in ("acc", "ece", "nll", "brier"):
        print(f"--- {m} ---")
        for c, v in d["corrupted"].items():
            vals = [round(v[f"severity_{s}"][m], 4) for s in (1, 2, 3, 4, 5)]
            print(f"{c:15s}", vals)
