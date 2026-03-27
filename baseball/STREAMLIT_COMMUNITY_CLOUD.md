# Streamlit Community Cloud Deployment

This app can be deployed to Streamlit Community Cloud, but the cloud-safe setup should use **The Odds API** for live odds.

## What works on Community Cloud

- Live MLB probable pitchers
- Live weather pulls
- The Odds API live HR odds
- Model scoring
- Best-bets and all-loaded-bets tables

## What should not be part of the Community Cloud deployment

- OddsShopper browser-session login via Selenium/Chrome
- Local credential files such as `apiKey.txt`, `oddshopperLogin.txt`, or `oddshopper_cookies.json`

OddsShopper in this repo is a local workflow. It depends on browser automation and session state that is not a good fit for Community Cloud.

## Required repo files

- `streamlit_app.py`
- `requirements.txt`
- `live_feature_cache/`
- `saved_models/`

## Secrets

In Streamlit Community Cloud app settings, add:

```toml
ODDS_API_KEY = "your_the_odds_api_key_here"
```

## Deployment settings

- Main file path: `streamlit_app.py`
- Recommended odds source in the app: `Odds API`
- Do not enable `Use browser session for OddsShopper`

## Notes

- The precomputed feature cache in `live_feature_cache/` needs to be present in the deployed repo if you want the app to score immediately.
- If you refresh that cache locally, push the updated cache files to GitHub so the deployed app sees them.
