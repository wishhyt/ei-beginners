"""Minimal behavior cloning trainer from trajectory JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_samples(dataset_path: Path) -> list[dict[str, str]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "llm_output" in payload:
        data = payload["llm_output"]
    elif isinstance(payload, list):
        data = payload
    else:
        data = []

    cleaned: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).upper()
        obj = str(item.get("object", ""))
        if action and obj:
            cleaned.append({"action": action, "object": obj})
    return cleaned


def train_bc(dataset: str, output_dir: str, epochs: int = 1) -> str:
    data = _load_samples(Path(dataset))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "bc_policy.json"

    action_counter_by_object: dict[str, Counter[str]] = defaultdict(Counter)
    for _ in range(max(epochs, 1)):
        for row in data:
            action_counter_by_object[row["object"]][row["action"]] += 1

    policy = {
        obj: counts.most_common(1)[0][0] if counts else "NOOP"
        for obj, counts in sorted(action_counter_by_object.items())
    }

    payload = {
        "algorithm": "behavior_cloning",
        "epochs": epochs,
        "samples": len(data),
        "policy": policy,
    }
    model_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="outputs/rl/bc")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    path = train_bc(args.dataset, args.output, epochs=args.epochs)
    print(path)


if __name__ == "__main__":
    main()
