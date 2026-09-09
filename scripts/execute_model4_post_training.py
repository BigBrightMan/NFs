#!/usr/bin/env python3
"""Execute a previously resolved Model 4 post-training config."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.pipeline import execute_post_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(execute_post_training_config(args.config), indent=2))


if __name__ == "__main__":
    main()
