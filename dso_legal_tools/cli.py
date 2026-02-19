from __future__ import annotations

import argparse
import os
from pathlib import Path

from dso_common.config import env_float, env_str
from dso_legal_tools.rule_extraction import process_legal_terms


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dso_legal_tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    t2r = sub.add_parser("terms-to-text-rules", help="Convert text legal terms JSON into normalized legal rules JSON (LLM + postprocessing)")
    t2r.add_argument("--input", required=True, help="Path to text legal terms json")
    t2r.add_argument("--output", default=str(Path("data") / "rules" / "text_rules.json"), help="Output rules path")
    t2r.add_argument("--delay", type=float, default=None, help="Delay between LLM calls")

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "terms-to-text-rules":
        provider = env_str("PARSER_PROVIDER", "fireworks") or "fireworks"
        api_key = env_str("PARSER_API_KEY")
        if not api_key:
            if provider == "openai":
                api_key = env_str("OPENAI_API_KEY")
            else:
                api_key = env_str("FIREWORKS_API_KEY")
        if not api_key:
            raise SystemExit("Missing PARSER_API_KEY (or OPENAI_API_KEY/FIREWORKS_API_KEY)")

        model = env_str("PARSER_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct") or "accounts/fireworks/models/llama-v3p1-70b-instruct"
        delay = args.delay if args.delay is not None else float(env_str("PARSER_DELAY", "0.5") or "0.5")

        process_legal_terms(
            input_path=args.input,
            output_path=args.output,
            delay_between_calls=delay,
            parser_api_key=api_key,
            parser_provider=provider,
            parser_model=model,
        )
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
