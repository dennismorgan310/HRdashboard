import argparse
import json
import time
from pathlib import Path

import requests
from requests.cookies import create_cookie
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


LOGIN_INFO_PATH = Path("oddshopperLogin.txt")
COOKIES_PATH = Path("oddshopper_cookies.json")
LOGIN_URL = "https://oddsshopper.us.auth0.com/u/login?state=hKFo2SAtOTNGb204VnlnVm9jQVhJQU5GMmVkM1FwN1hlR2V0RKFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDlPbkdVTlNScnZxWmdzQVlzenIwaGt3ZGsxelRQblB2o2NpZNkgak9KR1V3ZTFmU0hGelJ0RjJ6OXAyMnV5RWFWWjhXdHc"
ODDSSHOPPER_BASE = "https://www.oddsshopper.com"
CHROME_BINARY_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def load_login_info(path: Path = LOGIN_INFO_PATH) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing login info file: {path}")

    email = None
    password = None
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "email":
            email = value
        elif key == "password":
            password = value

    if not email or not password:
        raise ValueError("Could not parse Email and Password from oddshopperLogin.txt")

    return email, password


def save_cookies(cookies: list[dict], path: Path = COOKIES_PATH) -> None:
    path.write_text(json.dumps(cookies, indent=2))


def load_cookies(path: Path = COOKIES_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def build_authenticated_session(cookies_path: Path = COOKIES_PATH) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    for cookie in load_cookies(cookies_path):
        request_cookie = create_cookie(
            name=cookie.get("name"),
            value=cookie.get("value"),
            domain=cookie.get("domain"),
            path=cookie.get("path", "/"),
            secure=bool(cookie.get("secure", False)),
            expires=cookie.get("expiry"),
            rest={"HttpOnly": cookie.get("httpOnly", False)},
        )
        session.cookies.set_cookie(request_cookie)
    return session


def login_and_save_cookies(
    login_info_path: Path = LOGIN_INFO_PATH,
    cookies_path: Path = COOKIES_PATH,
    login_url: str = LOGIN_URL,
    wait_seconds: int = 45,
    headless: bool = True,
) -> None:
    email, password = load_login_info(login_info_path)

    options = ChromeOptions()
    if CHROME_BINARY_PATH.exists():
        options.binary_location = str(CHROME_BINARY_PATH)
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1600,1200")
    else:
        options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    driver.get(login_url)

    wait = WebDriverWait(driver, wait_seconds)
    try:
        email_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email'], input[name='username']"))
        )
        email_input.clear()
        email_input.send_keys(email)

        password_input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password'], input[name='password']"))
        )
        password_input.clear()
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)

        wait.until(lambda d: "oddsshopper.com" in d.current_url and "/api/auth/callback" not in d.current_url)
        time.sleep(3)
        cookies = driver.get_cookies()
        save_cookies(cookies, cookies_path)
    except TimeoutException as exc:
        raise RuntimeError(
            "Timed out during OddsShopper login. "
            "If the Auth0 page changed or Chrome requires extra verification, complete the login manually in the opened "
            "window and re-run this script."
        ) from exc
    finally:
        driver.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log in to OddsShopper via Chrome and save authenticated cookies."
    )
    parser.add_argument("--login-info-path", default=str(LOGIN_INFO_PATH))
    parser.add_argument("--cookies-path", default=str(COOKIES_PATH))
    parser.add_argument("--login-url", default=LOGIN_URL)
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chrome with a visible window instead of headless mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    login_and_save_cookies(
        login_info_path=Path(args.login_info_path),
        cookies_path=Path(args.cookies_path),
        login_url=args.login_url,
        headless=not args.headed,
    )
    print(f"Saved OddsShopper cookies to {args.cookies_path}")


if __name__ == "__main__":
    main()
