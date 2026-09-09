#!/usr/bin/env python3
"""Execute one resolved Model 4 generation config."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.generation import generate_model4_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(generate_model4_from_config(args.config), indent=2))


if __name__ == "__main__":
    main()
