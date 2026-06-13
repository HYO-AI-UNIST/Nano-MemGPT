from __future__ import annotations

import argparse
import os

from openai import APIError, AuthenticationError, OpenAI, RateLimitError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the configured OpenAI teacher can complete a minimal paid request."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_TEACHER_MODEL", "gpt-4-turbo-2024-04-09"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("status=blocked reason=missing_openai_api_key")
    try:
        response = OpenAI(api_key=api_key, max_retries=0).chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply only with OK."}],
            temperature=0,
            max_tokens=3,
        )
    except AuthenticationError as exc:
        raise SystemExit(f"status=blocked reason=authentication_error detail={exc}") from exc
    except RateLimitError as exc:
        raise SystemExit(f"status=blocked reason=rate_limit_or_insufficient_quota detail={exc}") from exc
    except APIError as exc:
        raise SystemExit(f"status=blocked reason=openai_api_error detail={exc}") from exc
    answer = response.choices[0].message.content or ""
    print(f"status=ready teacher_model={args.model} response={answer!r}")


if __name__ == "__main__":
    main()
