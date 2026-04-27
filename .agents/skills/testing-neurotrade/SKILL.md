# Testing NeuroTrade Streamlit App

## Overview
NeuroTrade is a Streamlit-based trading dashboard with a landing page gate. The app has two main views: a landing page (HTML embed) and the trading dashboard.

## Prerequisites
- Python 3.12+
- All dependencies installed: `pip install -r requirements.txt`
- **Important**: `matplotlib` is required but might be missing from requirements.txt. Install it manually if you see `ImportError: 'Import matplotlib' failed` on tabs like Data Explorer, AI Predictions, etc.

## Starting the App
```bash
cd /home/ubuntu/repos/NeuroTrade
streamlit run app.py --server.headless true --server.port 8501
```
The app will be available at `http://localhost:8501`.

## App Navigation Flow

### Landing Page (shown first)
1. The landing page is embedded HTML (`landingpage.html`) rendered via `st.components.v1.html()`
2. It contains: nav bar (NeuroTrade logo, About, Product, Resources, Learn More, Try Now), hero section, ticker marquee, features section, CTA section, footer
3. **To enter the dashboard**: Scroll to the very bottom of the page and click the Streamlit button **"⚡ Launch Trading Terminal"** (this is outside the HTML iframe, rendered by Streamlit at `app.py:180`)
4. The "Try Now" button inside the HTML landing page may not work for transitioning — use the Streamlit button instead

### Dashboard
1. After clicking Launch, `st.session_state.show_dashboard` is set to `True` and `st.rerun()` is called
2. The app auto-loads AAPL data on first dashboard load
3. **Sidebar sections** (with emoji headers): 📡 Market Data, 📊 Chart Overlays, 📈 Analysis Panels, ⚙️ RSI Sensitivity, 🧩 Available Tools
4. **Tab names** (user-friendly): 📊 Dashboard, 🤖 AI Predictions, 🧠 Neural Forecast, ⚛️ Quantum AI, 📈 Strategy Tester, 🔄 Market Mood, 🔥 Risk Simulator, 📰 Market News, 📋 Data Explorer
5. Tab availability depends on which Python modules are installed (shown with green/red dots in sidebar)

## Key Test Scenarios

### 1. Landing Page → Dashboard Transition
- Verify landing page loads first (nav bar, hero visible)
- Scroll to bottom, click "Launch Trading Terminal"
- Dashboard should load with AAPL data, no errors

### 2. UI Theme Verification
- Dark background (deep dark, near #05080f)
- Neon green (#00ffaa) accent colors on badges, buttons, chart elements
- Glassmorphism cards with semi-transparent backgrounds and rounded corners
- Glass panel header showing ticker, price, change %, signal badge

### 3. Data Loading
- Default: AAPL auto-loads on first visit
- Manual: Change ticker input → click "⚡ Analyze Market" button
- Verify header updates with new ticker name and price
- Verify metric cards (RSI, MACD, ADX, ATR, VWAP, CCI) update

### 4. Chart Interactivity
- Toggle SMA/EMA/BB/VWAP overlays on/off in sidebar → chart legend updates
- Hover over chart → tooltip shows date and values
- Chart has dark background with green spike lines

### 5. Tab Content
- Each tab has a heading, description, and action button
- Data Explorer shows a styled data table with Download CSV
- Some tabs may show errors if optional dependencies are missing

## Known Issues
- **matplotlib dependency**: Not listed in requirements.txt but required by `Styler.background_gradient()` used in Data Explorer and other tabs. Install with `pip install matplotlib`.
- **Inactive tab labels truncated**: Streamlit renders only emojis for non-active tabs when many tabs are present. Full text shows when tab is selected. This is a Streamlit rendering limitation.
- **Landing page iframe scroll**: The landing page is embedded in a 3000px iframe. Need to scroll significantly to reach the Streamlit "Launch Trading Terminal" button at the bottom.

## Devin Secrets Needed
No secrets required — the app uses public yfinance data and runs locally.
