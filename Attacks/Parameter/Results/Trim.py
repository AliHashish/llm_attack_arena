import json
from collections import Counter, defaultdict
from pathlib import Path

# ====== Hardcoded config ======
IN_FILE  = Path("Parameters_google_gemma-3n-E4B-it.json")
OUT_FILE = Path("trimmed.json")
OUT_FILE = Path("Parameters_google_gemma-3n-E4B-it.json")
# keep first N occurrences for each param key; drop entire item beyond N
TRIMS = {
    "presence": 90,
    "frequency": 90,   # example: keep only first 90 items that have "frequency"
}
# ==============================

def load_items(path: Path):
    with path.open("r") as f:
        return json.load(f)

def count_param_keys(items):
    c = Counter()
    for it in items:
        params = it.get("param") or {}
        if isinstance(params, dict):
            c.update(params.keys())
    return c

def trim_items(items, trims):
    seen = defaultdict(int)
    kept = []
    dropped_idx = []

    for idx, it in enumerate(items):
        params = it.get("param")
        if not isinstance(params, dict):
            kept.append(it)
            continue

        # Does this item contain any key we’re trimming?
        relevant_keys = [k for k in params.keys() if k in trims]
        if not relevant_keys:
            kept.append(it)
            continue

        # Increment seen counts for relevant keys and decide keep/drop
        drop = False
        for k in relevant_keys:
            seen[k] += 1
            if seen[k] > trims[k]:
                drop = True

        if drop:
            dropped_idx.append(idx)
        else:
            kept.append(it)

    return kept, dropped_idx, dict(seen)

def main():
    items = load_items(IN_FILE)
    trimmed = items
    dropped_idx = []

    # 1) Count occurrences of each param key
    counts = count_param_keys(items)
    print("Param key counts:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    # 2) Trim by removing WHOLE items after first N for each key
    trimmed, dropped_idx, seen = trim_items(items, TRIMS)

    # 3) Report
    print("\nTrimming summary (keep first N by file order):")
    for k, n_keep in TRIMS.items():
        had = counts.get(k, 0)
        kept_n = min(had, n_keep)
        removed_n = max(0, had - kept_n)
        print(f"  {k}: kept {kept_n} / {had}, removed {removed_n}")

    print(f"\nDropped {len(dropped_idx)} items at indices: {dropped_idx[:20]}{' ...' if len(dropped_idx)>20 else ''}")

    # 4) Write output
    with OUT_FILE.open("w") as f:
        json.dump(trimmed, f, indent=4)
    print(f"\nWrote {len(trimmed)} items to {OUT_FILE}")

if __name__ == "__main__":
    main()
