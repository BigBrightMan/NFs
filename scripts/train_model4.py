#!/usr/bin/env python3
"""Train one Model 4 run from one fully resolved config."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.training.model4 import train_model4_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train_model4_from_config(args.config), indent=2))


if __name__ == "__main__":
    main()
