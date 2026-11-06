# AetherEngine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NSE India](https://img.shields.io/badge/Exchange-NSE%20India-orange.svg)](https://www.nseindia.com/)
[![Feed: DhanHQ L3](https://img.shields.io/badge/Feed-DhanHQ%2020--Depth-purple.svg)](https://dhanhq.co/)

**AetherEngine** is an institutional-grade, low-latency order book intelligence and automated paper execution platform for National Stock Exchange of India (NSE) equities. It streams DhanHQ's binary Level-3 (20-depth) market feed, computes 15 microstructural order book features in real time, and forecasts 3-second price drift using a walk-forward LightGBM ensemble.

The engine includes an interactive **Paper Trading Lab** web portal that simulates institutional intraday execution with full NSE transaction cost modeling (brokerage, STT, turnover charges, GST) and dynamic order book queue slippage.

> **Project Documentation & Executive Briefing:**
> - Detailed Quantitative Report: [`AetherEngine_Comprehensive_System_Report.pdf`](AetherEngine_Comprehensive_System_Report.pdf)
> - Slide Presentation: [`AetherEngine_Project_Presentation.pptx`](AetherEngine_Project_Presentation.pptx)

---

## Key System Highlights

* **Wire-Speed Binary Decoding:** Asynchronous parsing of Dhan's binary 20-depth packets (12-byte header + 20 × 16-byte bid/ask levels) in under 50 microseconds.
* **Continuous Feature Engine:** Extracts 15 normalized microstructure signals on every book change, including 3-tier depth band imbalances, micro-price offsets, and spread in basis points.
* **Large-Scale Multi-Day Validation:** Trained and empirically validated across **5 full NSE market sessions** (October 30 – November 6, 2026), capturing **7,103,671 real order book rows** across 40 liquid large-cap symbols.
* **1,059-Tree LightGBM Model:** Multiclass boosted decision trees (DOWN, FLAT, UP) trained with strict chronological expanding windows and label purging to eliminate lookahead bias.
* **Proven Out-of-Sample Predictability:** Achieves **46.8% balanced accuracy** (vs 33.3% random baseline) and **60.4% to 66.7% directional win precision** at high-conviction thresholds ($\ge 0.75 - 0.80$).
* **Friction-Aware Paper Broker:** Realistically penalizes taker orders with $\frac{1}{4}$ spread crossing costs, depth participation volume caps (50 bps), and strict NSE fee drag.

---

## Empirical Benchmark & Alpha Attribution

### Out-of-Sample Performance (November 6 Session, 1.14M Labeled Rows)

| Probability Threshold | Signals Fired | Directional Win Precision | Target Horizon | Operational Character |
| :--- | :--- | :--- | :--- | :--- |
| **0.70** | 20,727 | 44.6% Long / 50.5% Short | 3 seconds | Active / Moderate market noise |
| **0.72** | 12,163 | 46.8% Long / 54.0% Short | 3 seconds | Improved signal-to-noise |
| **0.75** | 4,487 | **48.8% Long / 60.4% Short** | 3 seconds | **Recommended Production Floor** |
| **0.78** | 1,149 | **54.5% Long / 62.5% Short** | 3 seconds | **High-Conviction Institutional** |
| **0.80** | 307 | **61.9% Long / 66.7% Short** | 3 seconds | **Maximum Precision / Low Churn** |

### Top Predictive Microstructure Features (Model Gain %)

| Feature | Microstructural Formulation | Model Gain Attribution |
| :--- | :--- | :--- |
| `band_1_imbalance` | Immediate Top-of-Book Depth Ratio: $(Q_{b1} - Q_{a1}) / (Q_{b1} + Q_{a1})$ | **31.2%** |
| `obi_l1` | Level-1 Order Book Imbalance | **14.5%** |
| `spread_bps` | Bid-Ask Spread normalized by Mid-Price in Basis Points | **12.8%** |
| `micro_offset_bps` | Weighted Micro-Price deviation from Mid-Price (bps) | **8.5%** |
| `band_2_imbalance` | Ticks 2–5 Consolidated Depth Imbalance Ratio | **3.9%** |
| *Bands 2 & 3 Depths* | Cumulative Bid/Ask Volumes across Ticks 6–20 | **29.1%** |

---

## System Architecture

```
                 ┌────────────────────────────────────────────────────────┐
                 │                DhanHQ WebSocket Feed                  │
                 │              (Binary 20-Depth L3 Packets)              │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     src.feed.binary_parser│
                               │  (Header + 20 Bid/Ask Px) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  src.features.orderbook   │
                               │  (Latest 20 Depth Cache)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  src.features.calculator  │
                               │   (15 Microstructure Feats│
                               └──────┬─────────────┬──────┘
                                      │             │
                    ┌─────────────────┘             └──────────────────┐
                    ▼                                                  ▼
     ┌───────────────────────────────┐                  ┌───────────────────────────────┐
     │     src.pipeline.recorder     │                  │    src.inference.predictor    │
     │  (Atomic Compressed Parquet)  │                  │  (1,059-Tree LightGBM Engine) │
     └───────────────────────────────┘                  └──────────────┬────────────────┘
                                                                       │
                                        ┌──────────────────────────────┴──────────────────────────────┐
                                        ▼                                                             ▼
                         ┌─────────────────────────────┐                               ┌─────────────────────────────┐
                         │      Rich Live Dashboard    │                               │   src.paper.broker (Lab)    │
                         │   (Terminal Tick Monitor)   │                               │  (Web UI @ 127.0.0.1:8000)   │
                         └─────────────────────────────┘                               └─────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites
* Python 3.11 or Python 3.12
* A Dhan trading account with **Data APIs** enabled ([dhanhq.co](https://dhanhq.co/))

### 1. Clone & Create Environment
```bash
git clone https://github.com/your-username/AetherEngine.git
cd AetherEngine

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # Optional: for tests & report building
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and insert your Dhan credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here

# Up to 50 instruments per connection
DHAN_INSTRUMENTS=NSE_EQ:1333,NSE_EQ:11536,NSE_EQ:1594,NSE_EQ:2885,NSE_EQ:10999,NSE_EQ:317,NSE_EQ:15083,NSE_EQ:1922
```

---

## Operating Modes & Usage

### 1. Record Live Market Data
Stream real-time 20-depth order books and write compressed Parquet batches during market hours (09:15 – 15:30 IST):
```bash
python -m src.main --record --data-dir data
```
* Partitions are organized chronologically: `data/<IST_DATE>/part-<recv_ns>.parquet`.
* Safe shutdown via `Ctrl+C` cleanly flushes memory buffers to disk.

### 2. Train the Model
Fit a walk-forward purged LightGBM model on recorded Parquet sessions:
```bash
python -m src.pipeline.trainer --data data --model model.txt --signal-threshold 0.75
```
* `--horizon`: Forecast horizon in seconds (default `3.0` s).
* `--threshold-pct`: Minimum return threshold for UP/DOWN labels (default `0.01%`).
* `--split-time`: Intraday train/test split boundary when training on a single day (default `13:00` IST).

### 3. Run Live Engine with Signal Dashboard
Stream live data, evaluate predictions in real time, and display the Rich terminal monitor:
```bash
python -m src.main --model model.txt --threshold 0.75 --focus NSE_EQ:11536
```
To run without the terminal dashboard (lowest latency for server deployments):
```bash
python -m src.main --model model.txt --headless --log-file aether.log
```

### 4. Launch the Paper Trading Lab (Web Portal)
Launch the local web portal to test strategies with zero financial risk:
```bash
# Live paper trading + web portal
python -m src.main --portal --model model.txt

# Offline replay mode (no Dhan credentials or market hours needed):
python -m src.main --portal --no-feed --data-dir data
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## Paper Trading Lab Parameters

The paper broker enforces institutional market mechanics:

| Parameter | Default | Practical Purpose |
| :--- | :--- | :--- |
| **Starting Capital** | ₹1,00,000 | Initial simulated account equity |
| **Trade Size** | ₹50,000 | Capital deployed per entry |
| **Leverage** | 1× (up to 5×) | Position size capped at `capital × leverage` |
| **Signal Threshold** | 0.75 | Model probability required to trigger entry |
| **Time Exit** | 3.0 s | Force close position after $N$ seconds |
| **Exit on Opposite** | Enabled | Early exit when the model flips direction |
| **Dynamic Slippage** | Enabled | Taker orders walk the 20 levels; pays $\frac{1}{4}$ spread + queue impact |
| **NSE Charges** | Enabled | Brokerage (0.03% max ₹20), STT (0.025% on sells), NSE exchange charges (0.00297%), SEBI fees, GST |

---

## Directory Layout

```
AetherEngine/
├── src/
│   ├── feed/               # Binary WebSocket ingestion & packet unpacking
│   ├── features/           # Level-3 order book cache & 15 microstructure features
│   ├── inference/          # Low-latency LightGBM predictor
│   ├── pipeline/           # Parquet partition recorder & walk-forward trainer
│   ├── paper/              # Simulated broker, charge calculations, replay manager
│   ├── portal/             # FastAPI portal server & single-page HTML5 lab
│   └── main.py             # CLI entrypoint, asyncio loop, and Rich dashboard
├── tests/                  # Offline test suite (mock server, feature math, broker)
├── .env.example            # Template for Dhan credentials & symbol universe
├── .gitignore              # Clean rules (blocks credentials, parquets, pycache)
├── model.txt               # Trained 1,059-tree LightGBM model artifact
├── pytest.ini              # Test runner configuration
├── requirements.txt        # Runtime dependencies
└── requirements-dev.txt    # Testing, linting, and report generation tools
```

---

## Testing & Quality Assurance

AetherEngine includes an offline test harness that verifies binary decoding, feature math, order book reconstruction, and simulated broker mechanics against hand-calculated synthetic markets:

```bash
# Run complete test suite
pytest -v

# Run fast unit tests (parser, client, features)
pytest -m "not slow"
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
