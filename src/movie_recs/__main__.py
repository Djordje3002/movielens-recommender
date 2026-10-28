"""Run from the repository root: python -m movie_recs train|predict."""

import argparse
import json
from pathlib import Path

from .model import predict, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=Path("examples/sample.json"))
    args = parser.parse_args()
    if args.command == "train":
        train(args.root)
    else:
        payload = json.loads(args.input.read_text())
        print(json.dumps(predict(args.root, payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
