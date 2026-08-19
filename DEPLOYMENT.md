# 🚀 RiskLens v2.1.0 Enterprise Production Deployment Runbook

This runbook details the production deployment for **RiskLens**:
- **Backend API Service**: FastAPI inference server, Telegram Webhook receiver, authenticated telemetry, and automated daily active learning retraining on Render / Railway.
- **Frontend Dashboard**: Streamlit Enterprise Intelligence dashboard.
- **Storage Strategy**: Persistent Disk Volume (`Option A`) mounted at `/app/databases` for zero-latency, persistent SQLite `feedback.db` and `usage.db`.

---

## 📋 1. Architecture & Pre-Flight Checklist

```
                      [ Telegram / Users / Clients ]
                                    │
                         HTTPS (TLS 1.3 Strict)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             Render / Railway Reverse Proxy               │
       └────────────┬───────────────────────────────┬────────────┘
                    │                               │
                    ▼                               ▼
     ┌────────────────────────────┐  ┌────────────────────────────┐
     │   risklens-api (FastAPI)   │  │ risklens-dashboard (UI)    │
     │   • POST /predict          │  │   • Port: 8501             │
     │   • POST /telegram/webhook │  │   • Streamlit App          │
     │   • GET /analytics         │  │   • Verify / Analytics     │
     │   • APScheduler @ 02:00 UTC│  └────────────────────────────┘
     └──────────────┬─────────────┘
                    │
                    ▼
     ┌────────────────────────────┐
     │  Persistent Volume Disk    │
     │  Mount: /app/databases     │
     │  • feedback.db             │
     │  • usage.db                │
     └────────────────────────────┘
```

### Pre-Flight Verification
- [x] `.env` is listed in [`.gitignore`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/.gitignore) and was never committed to git history.
- [x] [`.env.example`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/.env.example) contains all environment variable keys with empty/placeholder values.
- [x] All 20 security audit controls + Webhook authenticity verification are active.
- [x] Dependency scan passed via [`scripts/security_scan.py`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/scripts/security_scan.py).

---

## 🔑 2. Required Production Secrets & Environment Variables

Configure these secrets in your platform dashboard (do **NOT** bake them into container images):

| Variable Name | Purpose | Example / Format |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot authentication token | `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ` |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token for webhook authenticity | `32+ char random hex / alphanumeric string` |
| `GOOGLE_FACTCHECK_API_KEY` | Google Fact Check Tools API Key | `AIzaSy...` |
| `SERPER_API_KEY` | Serper.dev Google Web Search API Key | `sec_...` |
| `RISKLENS_API_KEY` | Admin key to access `/analytics` & dashboard | `rk_live_...` |
| `USER_ID_SALT` | Salt for user ID HMAC-SHA256 pseudonymization | `random_secret_salt_string` |
| `DATABASE_DIR` | Mount path for persistent SQLite volume | `/app/databases` |
| `ENVIRONMENT` | Runtime mode | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 📦 3. Step-by-Step Deployment (Render Blueprint)

### Step 3.1: Connect Repository to Render
1. Navigate to **[dashboard.render.com](https://dashboard.render.com)** and click **New +** $\rightarrow$ **Blueprint**.
2. Select your `misinformation-risk-intelligence` repository.
3. Render will automatically parse [`render.yaml`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/render.yaml) and configure:
   - **`risklens-api`** (Docker Web Service, Port 8000)
   - **`risklens-dashboard`** (Docker Web Service, Port 8501)
   - **`risklens-storage`** (5 GB Persistent Disk mounted at `/app/databases`)

### Step 3.2: Enter Secret Environment Variables
In the Render Blueprint setup screen, fill in the values for:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`
- `GOOGLE_FACTCHECK_API_KEY`
- `SERPER_API_KEY`
- `RISKLENS_API_KEY`
- `USER_ID_SALT`

### Step 3.3: Deploy & Verify Health
Click **Apply**. Once the build finishes:
1. Verify the backend health endpoint:
   ```bash
   curl -f https://<YOUR_RENDER_BACKEND_URL>/health
   ```
   **Expected Response (HTTP 200)**:
   ```json
   {
     "status": "healthy",
     "calibrated_ensemble_loaded": true,
     "scheduler_active": true,
     "telegram_webhook_active": true
   }
   ```
2. Open `https://<YOUR_RENDER_DASHBOARD_URL>` in your browser to confirm the Streamlit interface is live.

---

## 🤖 4. Telegram Webhook Registration (`setWebhook`)

Once the backend is live with HTTPS, register the webhook with Telegram using the exact `secret_token` configured in your environment.

### Step 4.1: Register Webhook
Run the following `curl` command (replace `<YOUR_TELEGRAM_BOT_TOKEN>`, `<YOUR_BACKEND_DOMAIN>`, and `<YOUR_TELEGRAM_WEBHOOK_SECRET>`):

```bash
curl -X POST "https://api.telegram.org/bot<YOUR_TELEGRAM_BOT_TOKEN>/setWebhook" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://<YOUR_BACKEND_DOMAIN>/telegram/webhook",
       "secret_token": "<YOUR_TELEGRAM_WEBHOOK_SECRET>",
       "max_connections": 40,
       "drop_pending_updates": true
     }'
```

**Expected Response**:
```json
{
  "ok": true,
  "result": true,
  "description": "Webhook was set"
}
```

### Step 4.2: Verify Webhook Status
Verify that Telegram confirms the webhook is healthy:

```bash
curl -X GET "https://api.telegram.org/bot<YOUR_TELEGRAM_BOT_TOKEN>/getWebhookInfo"
```

**Expected Output**:
```json
{
  "ok": true,
  "result": {
    "url": "https://<YOUR_BACKEND_DOMAIN>/telegram/webhook",
    "has_custom_certificate": false,
    "pending_update_count": 0,
    "max_connections": 40,
    "ip_address": "..."
  }
}
```

---

## ⏰ 5. Verifying the Automated Retraining Cron

- **Mechanism**: The backend runs `APScheduler` inside the persistent FastAPI process.
- **Schedule**: Every day at **02:00:00 UTC** (`CronTrigger(hour=2, minute=0, timezone="UTC")`).
- **Condition**: Retrains the active learning queue when $\ge 500$ new user feedback samples have been collected.
- **How to verify**:
  Check the backend logs on Render / Railway:
  ```bash
  # In Render / Railway Log Viewer:
  APScheduler initialized: Daily retraining scheduled for 02:00 UTC.
  ```

---

## 🛡️ 6. Dependency Vulnerability Scans

To run the automated dependency vulnerability audit in staging or CI/CD:

```bash
python scripts/security_scan.py
```
Outputs report to `results/security_audit_report.json`.

---

## 🔄 7. Rollback & Disaster Recovery Procedure

If a deployed build introduces an unexpected issue:

1. **Instant Rollback via Dashboard**:
   - In Render: Go to `risklens-api` $\rightarrow$ **Deploys** $\rightarrow$ Click **Rollback** on the previous green commit.
   - In Railway: Go to the service $\rightarrow$ **Deployments** $\rightarrow$ Click **Redeploy** on the last healthy deployment.
2. **Database Integrity**:
   - Because SQLite files live on the mounted persistent volume `/app/databases`, rolling back container images does **NOT** wipe or modify `feedback.db` or `usage.db`.
3. **Webhook Fallback**:
   - If the webhook needs emergency temporary redirection back to local polling for maintenance:
     ```bash
     curl -X POST "https://api.telegram.org/bot<YOUR_TELEGRAM_BOT_TOKEN>/deleteWebhook"
     ```

---

## 📊 8. Production Monitoring, Error Tracking & Alerting

RiskLens Enterprise includes an integrated zero-overhead observability suite:

### 1. Error Tracking (Sentry with PII Scrubbing)
- **Setup**: Create a free Sentry project (Python / FastAPI) at [sentry.io](https://sentry.io) (5,000 free errors/month).
- Add `SENTRY_DSN=https://<key>@o0.ingest.sentry.io/<id>` to your Render/Railway environment variables.
- **Data Privacy**: The built-in `_scrub_sentry_event` hook automatically scrubs raw message text, `X-Telegram-Bot-Api-Secret-Token`, API keys, query parameters, and raw user IDs before dispatching events.

### 2. Instant Push Alerting (Telegram Admin Chat & Webhooks)
- **Telegram Push Channel (Zero Cost)**: Set `TELEGRAM_ADMIN_CHAT_ID` to your numeric user ID or an admin monitoring group ID. The bot token dispatches push alerts directly into your chat whenever:
  - An unhandled pipeline or inference exception occurs.
  - Active learning nightly retraining fails.
  - Model live accuracy drifts below `ACCURACY_ALERT_THRESHOLD` (default 75%).
  - Telegram webhook reports a delivery backlog (> 20 updates) or recent delivery errors.
- **Slack / Discord Webhook**: Set `ALERT_WEBHOOK_URL` to receive rich formatted embeds in your team channel.

### 3. Deep Health Probing (UptimeRobot / BetterStack)
- Configure an external uptime checker (e.g. [UptimeRobot](https://uptimerobot.com) or [BetterStack](https://betterstack.com) free tiers) to ping:
  ```http
  GET https://risklens-api.onrender.com/health
  ```
- **Active Deep Checks**: The endpoint executes active `SELECT 1` queries on SQLite databases, validates storage write access, checks model weights in memory, and confirms scheduler status. Returns HTTP 200 when fully healthy, or HTTP 503 on service degradation.

### 4. Background Automated Telemetry Jobs
- **Telegram Webhook Inspector**: Runs every 15 minutes via APScheduler, querying Telegram's `getWebhookInfo` to alert on dropped updates.
- **Daily Model Drift Inspector**: Runs at 01:00 UTC, comparing 7-day rolling live accuracy against the baseline alert threshold.
- **Operational Metrics**: Query `GET /operations/metrics?api_key=<KEY>` or view the **Observability** section directly in the Streamlit Analytics tab.

---

## 🤗 9. Hugging Face Spaces Deployment & GitHub-to-Space Sync

Hugging Face Spaces (`Preeti-Antil/RiskLens`) hosts the complete RiskLens ecosystem as a single multi-process Docker container, automatically mirrored from the GitHub source of truth on every push to `main`.

### Architecture Overview
- **Supervisor**: `entrypoint.sh` starts FastAPI on `127.0.0.1:8000`, the Telegram Bot worker in long-polling mode in the background, and Streamlit in the foreground on exposed port `7860`.
- **Telegram Connection Mode**: Uses long-polling (`TELEGRAM_MODE=polling`). When the Space is active or awakened by incoming traffic, the bot reconnects to Telegram and pulls all queued messages.
- **Persistent Storage**: When Persistent Storage is enabled in Space Settings, it mounts `/data`. RiskLens detects `/data` and sets `DATABASE_DIR=/data`, preserving `feedback.db` and `usage.db` across container rebuilds.
- **Scheduled Wake & Retraining**: `.github/workflows/retrain-cron.yml` executes daily at 02:00 UTC, sending a health probe to wake the Space and triggering active learning retraining.

### Step-by-Step Setup

#### Step 1: Generate a Hugging Face Write Token
1. Go to [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. Click **New token**.
3. Name: `RiskLens-GitHub-Sync`.
4. Type: Select **Write** (or **Fine-grained** with `Manage Spaces` permissions).
5. Click **Create token** and copy the resulting `hf_...` token.

#### Step 2: Add Secret to GitHub Repository
1. Navigate to your GitHub repository: `https://github.com/Preeti-Antil02/misinformation-risk-intelligence`.
2. Go to **Settings** → **Secrets and variables** → **Actions**.
3. Click **New repository secret**.
4. Name: `HF_TOKEN`
5. Value: Paste your Hugging Face write token.
6. (Optional) Add `RISKLENS_API_KEY` to enable the scheduled wake/retrain workflow.

#### Step 3: Configure Space Variables & Secrets on Hugging Face
In your Space Settings ([huggingface.co/spaces/Preeti-Antil/RiskLens/settings](https://huggingface.co/spaces/Preeti-Antil/RiskLens/settings)), scroll to **Variables and secrets** and add:

**Repository Secrets:**
- `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API token.
- `TELEGRAM_ADMIN_CHAT_ID`: Your Telegram numeric chat ID for alerts.
- `GOOGLE_FACTCHECK_API_KEY`: Google Fact Check Tools API key.
- `SERPER_API_KEY`: Serper Google Web Search API key.
- `RISKLENS_API_KEY`: Secret API key for admin routes.
- `USER_ID_SALT`: Salt string for HMAC user pseudonymization.
- `SENTRY_DSN`: (Optional) Sentry project DSN for error tracking.
- `ALERT_WEBHOOK_URL`: (Optional) Discord/Slack alert webhook.

**Repository Variables:**
- `ENVIRONMENT`: `production`
- `TELEGRAM_MODE`: `polling`
- `DATABASE_DIR`: `/data` (if Persistent Storage enabled) or `/app/databases` (ephemeral default).

#### Step 4: Verify the GitHub Action Sync
1. Push any commit to `main` on GitHub:
   ```bash
   git add .
   git commit -m "Deploy RiskLens v2.1.0 to Hugging Face Spaces"
   git push origin main
   ```
2. Navigate to the **Actions** tab on GitHub and inspect the **Sync GitHub to Hugging Face Space** workflow.
3. The workflow force-pushes `main` to `https://huggingface.co/spaces/Preeti-Antil/RiskLens`, triggering an automated Docker container build and deployment on Hugging Face Spaces.


