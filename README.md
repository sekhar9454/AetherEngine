# AetherEngine

Streams the DhanHQ 20-level market depth feed, computes order book features on every update, and predicts whether the mid-price will move up, down or stay flat over the next 3 seconds using LightGBM. Signals are shown on a live terminal dashboard. A local web portal, the **paper trading lab**, simulates trading those signals with fake money, either on recorded days or live. The engine never places real orders.

For a non-technical explanation of the idea, see `The Market Alpha Engine, Explained Simply.pdf`.

## Setup

Requires Python 3.11+ and a Dhan account with Data APIs enabled.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add requirements-dev.txt for tests
```

Fill in `.env` (it is git-ignored):

```ini
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token
# Comma-separated SEGMENT:SECURITY_ID pairs, max 50 per connection.
DHAN_INSTRUMENTS=NSE_EQ:1333
```

Segments: `IDX_I`, `NSE_EQ`, `NSE_FNO`, `NSE_CURRENCY`, `BSE_EQ`, `MCX_COMM`, `BSE_CURRENCY`, `BSE_FNO`.

## Usage

The workflow has three steps: record live data, train on it, then run live with the model. Run every command from this directory.

```bash
# 1. Record during market hours (09:15-15:30 IST). Ctrl+C flushes and exits.
python -m src.main --record

# 2. Train on everything recorded under data/
python -m src.pipeline.trainer --data data --model model.txt

# 3. Live signals (optionally keep recording at the same time)
python -m src.main --model model.txt --record
```

Running `python -m src.main` with neither flag just shows the dashboard. Add `--headless` to log a status line every 10 s instead, which is best for servers and lowest latency.

### `src.main` options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--record` | off | Write feature vectors and the raw book to Parquet |
| `--data-dir` | `data` | Where recordings go |
| `--flush-interval` | `60` | Max seconds between Parquet flushes (also flushes every 50,000 rows) |
| `--model` | none | LightGBM `model.txt` for live predictions |
| `--threshold` | `0.70` | Probability needed for a LONG or SHORT signal |
| `--focus` | first instrument | `SEGMENT:SECURITY_ID` shown on the dashboard |
| `--headless` | off | No dashboard; log status lines instead |
| `--refresh-hz` | `4` | Dashboard refresh rate |
| `--queue-size` | `10000` | Feed buffer; when full, the oldest update is dropped |
| `--log-file` | none | Also write logs to this file |

Exit codes: `0` normal shutdown, `1` Dhan rejected the session (bad or expired token, Data APIs not enabled, too many connections), `2` configuration error.

### `src.pipeline.trainer` options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data` | `data` | Recorder output directory |
| `--model` | `model.txt` | Output path |
| `--security-id` | all | Train on one instrument only (recommended when recording several) |
| `--horizon` | `3` | Label look-ahead in seconds |
| `--threshold-pct` | `0.01` | Move (in %) that counts as UP or DOWN |
| `--split-time` | `13:00` | IST train/test boundary when only one session is recorded |
| `--signal-threshold` | `0.70` | Threshold used when reporting signal precision |

The trainer prints out-of-sample results per fold, a confusion matrix, and feature importance. Read the **LONG prec** and **SHORT prec** columns: they show how often a signal at the chosen threshold was right on data the model never saw.

## Paper trading lab

A local web portal for trading the model's signals with fake money and comparing settings.

```bash
python -m src.main --portal                 # live feed + lab at http://127.0.0.1:8000
python -m src.main --portal --record        # also keep recording while you experiment
python -m src.main --portal --no-feed       # replay-only lab; no Dhan credentials or market hours needed
```

It has two modes. Both use the same simulated broker, so their results are directly comparable.

| Mode | Data | Time per experiment | Use it to |
| --- | --- | --- | --- |
| **Replay** | Days already recorded under `--data-dir` | Seconds | Try many settings quickly |
| **Live paper** | The running Dhan feed | Until you press Stop | Confirm the best settings on fresh data |

Live mode can only trade instruments listed in `DHAN_INSTRUMENTS`. Several live experiments can run side by side, each with its own model and settings. The portal lists every LightGBM model file (`*.txt`) in `--models-dir`, which is the current directory by default.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| Starting capital | ₹1,00,000 | Fake money at the start |
| Trade size | ₹50,000 per trade | Either ₹ per trade or a fixed number of shares |
| Leverage | 1× | Position value can be at most equity × leverage (up to 5×) |
| Allow short trades | on | Whether SHORT signals open short positions |
| Signal threshold | 0.70 | Probability needed for LONG or SHORT |
| Hold for | 3 s | Time exit; 0 turns it off |
| Exit on opposite signal | on | Close early when the model flips; it can reverse on the same update |
| Cooldown | 0 s | Wait after an exit before the next entry |
| Delay to fill | 0 ms | Time between a decision and its fill; the fill uses the book at that later moment |
| Start trading at / Square off at | 09:15 / 15:15 IST | No entries outside this window; open positions close at square-off |
| Daily loss limit | ₹2,000 | Close and stop for the rest of the day once the day's loss reaches this; 0 turns it off |
| Charges | on | Brokerage (0.03%, capped at ₹20 per order), STT 0.025% on sells, NSE fee 0.00297%, SEBI ₹10 per crore, stamp duty 0.003% on buys, and 18% GST on brokerage and fees. These approximate Dhan's NSE intraday equity charges; check them against Dhan's current pricing. |

### How fills are simulated

- **Price.** A buy fills against the ask side and a sell against the bid side. Each walks the 20 visible levels from the best price outward, so large orders get a worse average price.
- **One position at a time.** There is no pyramiding, and nothing is held overnight. If data ends before square-off, the position closes at the last price.
- **Equity.** Open positions are valued at the price they could close at right now (the best bid for longs, the best ask for shorts), not the mid-price.
- **Limitation.** Fills assume your order takes displayed liquidity instantly and does not move the market. Real fills are usually a little worse, so use the delay setting to stress-test.

### Results

Each experiment shows:
- net and gross P&L, charges, return, win rate, profit factor, maximum drawdown and average hold time;
- an equity curve and a per-day table;
- the most recent trades, with the full list as a CSV download.

Experiments are saved under `experiments/<id>/`: `experiment.json` holds the settings and results, and `trades.csv` holds every trade. They are reloaded on restart. Live experiments are also saved every 30 s. On shutdown (Ctrl+C), live experiments are squared off and saved before the feed closes.

The portal listens on 127.0.0.1 only. The process holds your Dhan token, so do not expose `--host` to a network. API docs are at `/api/docs`.

### Lab options for `src.main`

| Flag | Default | Purpose |
| --- | --- | --- |
| `--portal` | off | Serve the lab |
| `--host` / `--port` | `127.0.0.1` / `8000` | Where it listens |
| `--no-feed` | off | Run the lab without connecting to Dhan (replay only) |
| `--experiments-dir` | `experiments` | Where experiments are saved |
| `--models-dir` | `.` | Where the lab looks for model files |

## How it works

```
Dhan WebSocket ─▶ binary_parser ─▶ queue ─▶ order book ─▶ features ─┬─▶ recorder (Parquet)
                                                                   └─▶ predictor ─▶ signal ─▶ dashboard
```

| Module | Responsibility |
| --- | --- |
| `src/feed/websocket_client.py` | Connects, subscribes (RequestCode 23), reconnects with jittered exponential backoff, sends RequestCode 12 on shutdown |
| `src/feed/binary_parser.py` | Splits stacked packets in each frame and decodes the 12-byte header plus 20 × 16-byte levels |
| `src/features/orderbook.py` | Holds the latest 20 bid and ask levels per instrument |
| `src/features/calculator.py` | Computes the feature vector (below) |
| `src/pipeline/recorder.py` | Buffers rows; writes atomic part files to `data/<IST date>/part-<recv_ns>.parquet` |
| `src/pipeline/trainer.py` | Labels, walk-forward validation, final LightGBM fit |
| `src/inference/predictor.py` | Loads `model.txt` and returns P(DOWN), P(FLAT), P(UP) and the signal |
| `src/main.py` | Orchestration, batching, signal edge detection, Rich dashboard |
| `src/paper/broker.py` | Simulated broker: fills, charges, exits and risk limits, driven only by update timestamps |
| `src/paper/replay.py` | Loads recorded days and feeds them through the broker |
| `src/paper/manager.py` | Starts, stops, saves and reloads experiments; connects live experiments to the engine |
| `src/portal/server.py` | FastAPI app and uvicorn server running in the engine's event loop |
| `src/portal/static/index.html` | The lab's single page (plain HTML and JavaScript, no build step) |

**Features.** Spread (absolute and bps), mid-price, micro-price and its offset from mid, order book imbalance over the top 1, 5 and 20 levels, and bid/ask volume within 0–0.05%, 0.05–0.15% and 0.15–0.50% of mid, plus the imbalance of each band. Mid-price and micro-price are recorded but not fed to the model, because absolute price levels drift between days.

**Labels.** Each row gets the mid-price at the first tick at or after t + 3 s, within the same session. The row is dropped if that tick arrives more than 2 s late, for example after a feed gap. A move above +0.01% is UP (1), below −0.01% is DOWN (−1), otherwise FLAT (0).

**Validation.** The split is always chronological, never random. With one recorded session, the model trains before 13:00 and tests after. With several, each session is tested on a model trained only on earlier sessions. Training rows whose label window reaches into the test period are purged. The final model is fit on all data, using the median best iteration across folds.

**Batching.** The engine applies every queued packet before computing features, then computes once per changed book. A frame's bid and ask are therefore never evaluated half-applied, and a backlog collapses to the latest state instead of being replayed.

## Data layout

Each Parquet row holds `recv_ns` (receive time in epoch ns; Dhan packets carry no exchange timestamp), `exchange_segment`, `security_id`, every feature column, and the raw book as `bid_px`, `bid_qty`, `ask_px`, `ask_qty` (fixed-size arrays of 20). Keeping the raw book lets you design new features later without re-recording.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                   # full suite, about 30 s
pytest -m "not slow"     # parser, client and feature tests, about 1 s
```

The suite runs entirely offline. It uses a local mock of the Dhan server and a synthetic market (`tests/synth.py`) in which book imbalance genuinely predicts price drift. It checks:

- packet decoding;
- reconnect and fatal-error handling;
- feature math against hand-worked books;
- labels against brute force;
- that the model beats chance on an unseen day;
- graceful SIGINT shutdown of the real CLI;
- the paper broker against hand-worked books (fills, charges, every exit rule and risk limit);
- replay and live experiments through the portal API, and the portal CLI itself.

## Known caveats

- **Not yet verified against the live Dhan feed.** Everything has been tested against a mock built from Dhan's documentation. Dhan's docs place the bid/ask response code at byte 3 in one table and byte 12 in another; this code follows the header table (byte 3). If the first live run shows no updates or nonsense prices, check that first.
- **The subscribe code is 23.** RequestCode 15 is the ticker code of the regular market feed and does not apply to the 20-depth endpoint.
- **Latency depends on the machine.** Median tick-to-signal measured 0.3–0.8 ms on a laptop using the `powersave` CPU governor, with a p99 of 2–3 ms. Use the `performance` governor and `--headless` for the best numbers. The dashboard redraws on the same event loop as the feed.
- **Signals can flicker.** When a probability hovers near the threshold, the signal flips between LONG and FLAT. Hysteresis (separate entry and exit thresholds) is a possible improvement.
- **Synthetic results say nothing about real alpha.** Record several real sessions before trusting any model. A 0.01% move is often smaller than trading costs.
