#!/usr/bin/env python3
"""Print the resumable state stored in a training checkpoint."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.training import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
