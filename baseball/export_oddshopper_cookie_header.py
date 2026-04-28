import argparse
from pathlib import Path

from oddshopper_auth import COOKIES_PATH, load_cookies


def build_cookie_header(cookies_path: Path = COOKIES_PATH) -> str:
    parts = []
    for cookie in load_cookies(cookies_path):
        name = str(cookie.get("name") or "").strip()
        value = str(cookie.get("value") or "").strip()
        if name and value:
            parts.append(f"{name}={value}")
    if not parts:
        raise ValueError(f"No cookies found in {cookies_path}")
    return "; ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export saved OddsShopper Selenium cookies as an ODDSHOPPER_COOKIE_HEADER value."
    )
    parser.add_argument("--cookies-path", default=str(COOKIES_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cookie_header = build_cookie_header(Path(args.cookies_path))
    print("ODDSHOPPER_COOKIE_HEADER=")
    print(cookie_header)


if __name__ == "__main__":
    main()
