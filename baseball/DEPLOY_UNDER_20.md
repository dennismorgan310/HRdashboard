# Under-$20 Mobile Deployment

Recommended first target: Railway Hobby plus GitHub Actions.

Railway is a good fit for this dashboard because it can run the Streamlit web service from the Dockerfile, store secrets as environment variables, and stay under the target budget if we keep the app to one small service. GitHub Actions can refresh the small live feature cache on a schedule, so the laptop can stay off.

## Expected Cost

- Railway Hobby subscription: about $5/month.
- App compute/storage usage: commonly another $0-$15/month for a small Streamlit app.
- Target budget: keep the service at 1-2 GB RAM and watch usage for the first week.

If cloud Selenium/Chrome becomes required for automatic OddsShopper re-login, expect the cost and complexity to rise. The under-$20 version should use a server-side `ODDSHOPPER_COOKIE_HEADER` secret.

## Files That Must Be Deployed

- `streamlit_app.py`
- `requirements.txt`
- `Dockerfile`
- `live_feature_cache/latest_batter_features.parquet`
- `live_feature_cache/latest_pitcher_features.parquet`
- `live_feature_cache/latest_pitcher_split_features.parquet`
- `live_feature_cache/metadata.json`
- `saved_models/residual_late_snapshot_champion.pkl`

Do not deploy or commit:

- `apiKey.txt`
- `oddshopperLogin.txt`
- `oddshopper_cookies.json`
- `oddshopper_cookie_header.txt`
- `.streamlit/secrets.toml`
- `live_feature_cache/statcast_history.parquet`

## Railway Setup

1. Push this repo to GitHub.
2. Create a Railway project from the GitHub repo.
3. Railway should detect and build the root `Dockerfile`.
4. Optional: set the project root directory to `baseball` if you want Railway to use `baseball/Dockerfile` instead.
5. Add environment variables:

```text
ODDSHOPPER_COOKIE_HEADER=your_cookie_header_here
ODDS_API_KEY=optional_fallback_key
```

6. Deploy and open the generated Railway URL on your phone.

## OddsShopper Cookie Refresh

When the OddsShopper cookie expires, refresh it locally:

```bash
cd baseball
python3 oddshopper_auth.py --headed
python3 export_oddshopper_cookie_header.py
```

Paste the printed cookie value into Railway's `ODDSHOPPER_COOKIE_HEADER` environment variable and redeploy/restart the service.

## Cache Refresh

The repo includes `.github/workflows/refresh_live_feature_cache.yml`, which can refresh the small latest feature cache files daily or on demand.

To run it manually:

1. Open the GitHub repo.
2. Go to Actions.
3. Choose `Refresh live feature cache`.
4. Click `Run workflow`.

The workflow commits these files back to the repo:

- `live_feature_cache/latest_batter_features.parquet`
- `live_feature_cache/latest_pitcher_features.parquet`
- `live_feature_cache/latest_pitcher_split_features.parquet`
- `live_feature_cache/metadata.json`

Railway should redeploy after that commit, giving the phone app a fresh cache without the laptop.

You can still refresh locally when needed:

```bash
cd baseball
python3 build_live_feature_cache.py --date YYYY-MM-DD
git add live_feature_cache/latest_batter_features.parquet live_feature_cache/latest_pitcher_features.parquet live_feature_cache/latest_pitcher_split_features.parquet live_feature_cache/metadata.json
git commit -m "Refresh live feature cache"
git push
```
