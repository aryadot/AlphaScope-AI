"""
AlphaScope AI — Full-Stack Stock Analysis Engine
================================================
Three-layer analysis: Fundamentals (yfinance) + Technicals (pandas) + Sentiment (FinBERT)
Narrative generation & RAG chat via Groq (Llama 3.3 70B)
Built with: Streamlit · yfinance · HuggingFace Transformers · Groq
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEBUG SETTINGS - Toggle these as needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEBUG_MODE = True  # Set to False to hide debug output in production
if DEBUG_MODE:
    import sys
    print("="*60)
    print("AlphaScope AI - Debug Mode Active")
    print("="*60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dividend Yield Helper Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def safe_dividend_yield(info, ticker):
    """
    Safely calculate dividend yield with fallbacks and extensive debugging.
    Yahoo Finance can be inconsistent - this handles multiple cases.
    """
    if DEBUG_MODE:
        print(f"\n🔍 [DIVIDEND DEBUG] Processing {ticker}")
    
    # Try direct yield field first
    raw_yield = info.get("dividendYield")
    if DEBUG_MODE:
        print(f"  dividendYield raw: {raw_yield} (type: {type(raw_yield)})")
    
    # If it exists and seems reasonable
    if raw_yield is not None:
        # If it's > 1, it's probably already a percentage (like 39.0 meaning 39%)
        if raw_yield > 1:
            if DEBUG_MODE:
                print(f"  → Value > 1, converting to decimal: {raw_yield} → {raw_yield / 100}")
            result = raw_yield / 100
        # If it's between 0 and 1, it's already decimal
        elif 0 <= raw_yield <= 1:
            if DEBUG_MODE:
                print(f"  → Value between 0-1, using as-is: {raw_yield}")
            result = raw_yield
        else:
            result = None
            
        # SANITY CHECK: If result > 0.15 (15% yield), it's almost certainly wrong
        # Most stocks don't yield above 5-6%, 15% is bankruptcy territory
        if result and result > 0.15:
            if DEBUG_MODE:
                print(f"  ⚠️ Sanity check: {result*100:.1f}% yield is implausibly high, checking alternatives...")
            
            # Try trailing yield as fallback
            trail_yield = info.get("trailingAnnualDividendYield")
            if DEBUG_MODE:
                print(f"  trailingAnnualDividendYield: {trail_yield}")
            
            if trail_yield and 0 < trail_yield <= 0.15:
                if DEBUG_MODE:
                    print(f"  → Using trailing yield instead: {trail_yield}")
                return trail_yield
            
            # Try calculating from dividendRate
            rate = info.get("dividendRate")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if DEBUG_MODE:
                print(f"  dividendRate: {rate}, price: {price}")
            
            if rate and price and price > 0:
                # If rate is quarterly (most US stocks)
                if rate < 10:  # Sanity check - quarterly dividend under $10
                    calculated = (rate * 4) / price
                    if DEBUG_MODE:
                        print(f"  → Calculated from rate: ({rate} * 4) / {price} = {calculated}")
                    if calculated <= 0.15:
                        return calculated
            
            # If we got here, all fallbacks failed or also gave high values
            if DEBUG_MODE:
                print(f"  ⚠️ All fallbacks implausible, capping at 5% as safety")
            return 0.05  # Cap at 5% as absolute maximum reasonable yield
        
        return result
    
    # Try trailing annual dividend yield
    trail_yield = info.get("trailingAnnualDividendYield")
    if DEBUG_MODE:
        print(f"  trailingAnnualDividendYield: {trail_yield}")
    
    if trail_yield is not None:
        if trail_yield > 1:
            result = trail_yield / 100
        else:
            result = trail_yield
        
        if result <= 0.15:
            return result
    
    # Try calculating from dividendRate
    rate = info.get("dividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if DEBUG_MODE:
        print(f"  dividendRate: {rate}")
        print(f"  currentPrice: {price}")
    
    if rate and price and price > 0:
        # If rate is quarterly (most US stocks)
        if rate < 10:  # Sanity check - quarterly dividend under $10
            annual_dividend = rate * 4
            calculated = annual_dividend / price
            if DEBUG_MODE:
                print(f"  → Calculated from rate: ({rate} * 4) / {price} = {calculated}")
            if calculated <= 0.15:
                return calculated
    
    # Try last dividend value
    last_div = info.get("lastDividendValue")
    if DEBUG_MODE:
        print(f"  lastDividendValue: {last_div}")
    
    if last_div and price and price > 0:
        # Estimate annual from last dividend (assuming quarterly)
        estimated = (last_div * 4) / price
        if DEBUG_MODE:
            print(f"  → Estimated from last dividend: ({last_div} * 4) / {price} = {estimated}")
        if estimated <= 0.15:
            return estimated
    
    if DEBUG_MODE:
        print("  ❌ No reliable dividend data found, returning None")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Popular Tickers for Autocomplete
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POPULAR_TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "GOOG": "Alphabet Inc. (Class C)",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "MA": "Mastercard Inc.",
    "PG": "Procter & Gamble Co.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "The Home Depot Inc.",
    "DIS": "The Walt Disney Company",
    "BAC": "Bank of America Corp.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce Inc.",
    "NFLX": "Netflix Inc.",
    "XOM": "Exxon Mobil Corporation",
    "COST": "Costco Wholesale Corp.",
    "PEP": "PepsiCo Inc.",
    "KO": "The Coca-Cola Company",
    "CSCO": "Cisco Systems Inc.",
    "ABT": "Abbott Laboratories",
    "TMO": "Thermo Fisher Scientific",
    "MRK": "Merck & Co. Inc.",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    "ORCL": "Oracle Corporation",
    "NKE": "Nike Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "T": "AT&T Inc.",
    "VZ": "Verizon Communications",
    "UBER": "Uber Technologies Inc.",
    "SQ": "Block Inc. (Square)",
    "SPOT": "Spotify Technology",
    "SNAP": "Snap Inc.",
    "PLTR": "Palantir Technologies",
    "COIN": "Coinbase Global Inc.",
    "RIVN": "Rivian Automotive Inc.",
    "SOFI": "SoFi Technologies Inc.",
    "GME": "GameStop Corp.",
    "AMC": "AMC Entertainment",
    "BA": "The Boeing Company",
    "GS": "Goldman Sachs Group",
    "MS": "Morgan Stanley",
    "C": "Citigroup Inc.",
    "WFC": "Wells Fargo & Company",
    "ABNB": "Airbnb Inc.",
    "SHOP": "Shopify Inc.",
    "ZM": "Zoom Video Communications",
    "ROKU": "Roku Inc.",
    "DKNG": "DraftKings Inc.",
    "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike Holdings",
    "SNOW": "Snowflake Inc.",
    "NET": "Cloudflare Inc.",
    "MELI": "MercadoLibre Inc.",
    "SE": "Sea Limited",
    "BABA": "Alibaba Group",
    "TSM": "Taiwan Semiconductor",
    "SONY": "Sony Group Corporation",
    "TM": "Toyota Motor Corp.",
    "LLY": "Eli Lilly and Company",
    "AVGO": "Broadcom Inc.",
    "NOW": "ServiceNow Inc.",
}


def search_tickers(query: str) -> list[str]:
    """Return matching tickers + company names for autocomplete."""
    if not query:
        return []
    query = query.upper().strip()
    matches = []
    for ticker, name in POPULAR_TICKERS.items():
        if ticker.startswith(query) or query.lower() in name.lower():
            matches.append(f"{ticker} — {name}")
    # Sort: exact ticker prefix first, then alphabetical
    matches.sort(key=lambda x: (0 if x.split(" — ")[0].startswith(query) else 1, x))
    return matches[:8]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AlphaScope AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1rem;
    }

    .score-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .score-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .score-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .score-positive { color: #2ea043; }
    .score-negative { color: #f85149; }
    .score-neutral { color: #d29922; }

    .rec-badge {
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 2px solid;
        backdrop-filter: blur(10px);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .rec-badge.buy, .rec-badge.strong-buy { border-color: #2ea043; background: rgba(46,160,67,0.06); }
    .rec-badge.sell, .rec-badge.strong-sell, .rec-badge.strong-avoid, .rec-badge.avoid { border-color: #f85149; background: rgba(248,81,73,0.06); }
    .rec-badge.hold, .rec-badge.neutral, .rec-badge.wait, .rec-badge.caution { border-color: #d29922; background: rgba(210,153,34,0.06); }
    .rec-badge h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .rec-badge.buy h2, .rec-badge.strong-buy h2 { color: #2ea043; text-shadow: 0 0 30px rgba(46,160,67,0.3); }
    .rec-badge.sell h2, .rec-badge.strong-sell h2, .rec-badge.strong-avoid h2, .rec-badge.avoid h2 { color: #f85149; text-shadow: 0 0 30px rgba(248,81,73,0.3); }
    .rec-badge.hold h2, .rec-badge.neutral h2, .rec-badge.wait h2, .rec-badge.caution h2 { color: #d29922; text-shadow: 0 0 30px rgba(210,153,34,0.3); }
    .rec-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.5rem;
    }
    .rec-summary {
        font-size: 0.78rem;
        color: rgba(230,237,243,0.6);
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .rec-reason {
        font-size: 0.72rem;
        color: rgba(139,148,158,0.7);
        padding: 0.15rem 0;
    }
    .rec-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8b949e;
        background: rgba(0,0,0,0.3);
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .glass-panel {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .glass-panel h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .news-item {
        background: rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .news-headline {
        color: #e6edf3;
        font-size: 0.85rem;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 0.4rem;
    }
    .news-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .news-source { color: #8b949e; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
    .sentiment-tag {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }
    .sentiment-positive { background: rgba(46,160,67,0.15); color: #2ea043; }
    .sentiment-negative { background: rgba(248,81,73,0.15); color: #f85149; }
    .sentiment-neutral { background: rgba(139,148,158,0.15); color: #8b949e; }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-label { color: #8b949e; font-size: 0.85rem; }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #e6edf3;
        font-size: 0.9rem;
    }

    .disclaimer {
        text-align: center;
        color: rgba(139,148,158,0.35);
        font-size: 0.72rem;
        padding: 2.5rem 0 1rem 0;
        max-width: 650px;
        margin: 0 auto;
    }

    /* Hide streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Style the search input */
    .stTextInput > div > div > input {
        background-color: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e6edf3 !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(88,166,255,0.5) !important;
        box-shadow: 0 0 0 1px rgba(88,166,255,0.3) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(139,148,158,0.6) !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 3: FinBERT Sentiment Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Loading FinBERT sentiment model...")
def load_finbert():
    """Load ProsusAI/finbert from HuggingFace. Runs locally — no API key needed."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def analyze_sentiment(headlines: list[str]) -> list[dict]:
    """
    Run FinBERT on a list of financial news headlines.
    Returns list of {headline, sentiment, confidence, score} dicts.
    Score is mapped to -100..+100 scale.
    """
    if not headlines:
        return []
    try:
        classifier = load_finbert()
        results = classifier(headlines, truncation=True, max_length=512)
        scored = []
        for headline, result in zip(headlines, results):
            label = result["label"].lower()  # positive | negative | neutral
            conf = result["score"]            # 0.0 to 1.0

            if label == "positive":
                numeric = int(conf * 100)
            elif label == "negative":
                numeric = int(conf * -100)
            else:
                numeric = 0

            scored.append({
                "headline": headline,
                "sentiment": label,
                "confidence": round(conf, 3),
                "score": numeric,
            })
        return scored
    except Exception as e:
        st.warning(f"FinBERT failed: {e}")
        return [{"headline": h, "sentiment": "neutral", "confidence": 0.0, "score": 0} for h in headlines]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Groq API: Narrative Generation & RAG Chat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_narrative(ticker: str, fundamentals: dict, technicals: dict,
                       sentiment_score: int, composite: int, recommendation: str) -> str:
    """Use Groq (Llama 3 70B) to generate an analyst-style narrative grounded in the computed data."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_narrative(ticker, fundamentals, technicals, sentiment_score, composite, recommendation)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are a professional equity research analyst. Given the following data for {ticker}, write a 3-4 paragraph analysis covering:
1) Fundamental outlook — reference specific metrics
2) Technical picture — reference price, SMAs, RSI
3) Market sentiment from recent news
4) Summary verdict with key risks

Be specific with numbers. Professional but accessible tone. Do NOT give investment advice — frame as analysis only.

Fundamentals: {json.dumps({k: v for k, v in fundamentals.items() if v is not None}, default=str)}
Technicals: Price=${technicals.get('price', 'N/A'):.2f}, SMA50={technicals.get('sma50', 'N/A')}, SMA200={technicals.get('sma200', 'N/A')}, RSI={technicals.get('rsi', 'N/A'):.1f}
Sentiment Score: {sentiment_score}/100
Composite: {composite} → {recommendation}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        st.warning(f"Groq API error: {e}")
        return _fallback_narrative(ticker, fundamentals, technicals, sentiment_score, composite, recommendation)


def _fallback_narrative(ticker, fundamentals, technicals, sentiment_score, composite, recommendation):
    """Basic narrative when no API key is available."""
    pe = fundamentals.get("peRatio")
    pe_str = f"{pe:.1f}" if pe else "N/A"
    rg = fundamentals.get("revenueGrowth")
    rg_str = f"{rg*100:.1f}%" if rg else "N/A"
    price = technicals.get("price", 0)
    rsi = technicals.get("rsi", 0)
    sma50 = technicals.get("sma50")
    sma200 = technicals.get("sma200")

    trend = "above" if sma200 and price > sma200 else "below"

    return f"""**Fundamental Overview:** {ticker} trades at a trailing P/E of {pe_str} with revenue growth of {rg_str}. \
{"Valuation appears reasonable relative to growth." if pe and pe < 25 else "The elevated P/E suggests the market is pricing in significant future growth." if pe else "Limited fundamental data is available for a complete valuation picture."}

**Technical Picture:** The stock closed at ${price:.2f}, trading {trend} its 200-day moving average\
{f' (${sma200:.2f})' if sma200 else ''}. RSI sits at {rsi:.1f}\
{', indicating overbought conditions that may precede a pullback' if rsi > 70 else ', suggesting oversold conditions with potential for a bounce' if rsi < 30 else ', in neutral territory'}. \
{f'The 50-day SMA (${sma50:.2f}) is ' + ('above' if sma50 > sma200 else 'below') + ' the 200-day, ' + ('confirming bullish momentum (golden cross pattern).' if sma50 > sma200 else 'suggesting bearish momentum (death cross pattern).') if sma50 and sma200 else ''}

**Sentiment & Verdict:** Aggregate news sentiment scores {sentiment_score} on a -100 to +100 scale. Combined with fundamentals and technicals, the composite score of {composite} generates a **{recommendation}** signal. This is algorithmic analysis for informational purposes only — not investment advice.

_Set your `GROQ_API_KEY` environment variable for a richer AI-generated narrative._"""


def chat_with_groq(ticker: str, question: str, context: dict) -> str:
    """RAG-style chat grounded in the analysis data, powered by Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "💡 Chat requires a Groq API key. Set `GROQ_API_KEY` in your environment to enable the AI assistant."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        system_msg = f"""You are AlphaScope AI, a financial analysis assistant. You have access to the following analysis data for {ticker}. 
Answer questions using ONLY this data. Be concise, reference specific numbers, and say so if the data doesn't contain the answer.
Never give investment advice — you provide analysis only.

Analysis Data:
{json.dumps(context, default=str, indent=2)}"""

        # Build conversation history from session state
        messages = [{"role": "system", "content": system_msg}]
        if "chat_history" in st.session_state:
            for msg in st.session_state.chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.5,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Formatting Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fmt_large_number(n):
    if n is None: return "N/A"
    n = float(n)
    if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9: return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"

def fmt_pct(v):
    if v is None: return "N/A"
    # Sanity check - if it's > 5, it's probably already a percentage
    # (very few stocks yield > 5%)
    if v > 5:
        if DEBUG_MODE:
            print(f"  ⚠️ fmt_pct got value >5: {v} - treating as already percentage")
        return f"{v:.1f}%"
    # Normal case: decimal to percentage
    return f"{v*100:.2f}%"

def fmt_ratio(v):
    if v is None: return "N/A"
    return f"{v:.2f}"

def fmt_price(v):
    if v is None: return "N/A"
    return f"${v:.2f}"

def score_color_class(score):
    if score > 30: return "score-positive"
    if score < -30: return "score-negative"
    return "score-neutral"

def score_prefix(score):
    return f"+{score}" if score > 0 else str(score)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Fetching & Three-Layer Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str) -> dict:
    """
    Full AlphaScope analysis pipeline:
      Layer 1: Fundamental analysis from Yahoo Finance
      Layer 2: Technical indicators computed with pandas
      Layer 3: Sentiment analysis via FinBERT on news headlines
      Composite scoring & recommendation
    """

    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"🔍 RUN_ANALYSIS START: {ticker}")
        print(f"{'='*60}")

    stock = yf.Ticker(ticker)

    # ── LAYER 1: FUNDAMENTALS ──────────────────
    info = stock.info
    if not info or not info.get("regularMarketPrice"):
        return {"error": "Invalid ticker or no data available."}

    # Debug: Print all dividend-related fields
    if DEBUG_MODE:
        print("\n📊 [YAHOO DEBUG] All dividend-related fields:")
        div_fields = []
        for key, value in info.items():
            if any(x in key.lower() for x in ['dividend', 'yield', 'div', 'payout']):
                div_fields.append((key, value, type(value)))
        
        for key, value, typ in sorted(div_fields):
            print(f"  {key}: {value} ({typ})")
        
        print(f"\n  Current price: {info.get('currentPrice') or info.get('regularMarketPrice')}")

    fundamentals = {
        "peRatio": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
        "marketCap": info.get("marketCap"),
        "revenue": info.get("totalRevenue"),
        "revenueGrowth": info.get("revenueGrowth"),
        "eps": info.get("trailingEps"),
        "profitMargin": info.get("profitMargins"),
        "debtToEquity": info.get("debtToEquity"),
        "returnOnEquity": info.get("returnOnEquity"),
        "freeCashflow": info.get("freeCashflow"),
        "dividendYield": safe_dividend_yield(info, ticker),  # Using our safe function
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "shortName": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
    }

    if DEBUG_MODE:
        print(f"\n✅ [PROCESSED] dividendYield after safe_dividend_yield: {fundamentals['dividendYield']}")

    # Score fundamentals: -100 to +100
    f_score, f_count = 0, 0

    pe = fundamentals["peRatio"]
    if pe:
        f_count += 1
        f_score += 100 if pe < 15 else (0 if pe <= 25 else -100)

    rg = fundamentals["revenueGrowth"]
    if rg is not None:
        f_count += 1
        f_score += 100 if rg > 0.1 else (50 if rg > 0 else -100)

    de = fundamentals["debtToEquity"]
    if de is not None:
        f_count += 1
        f_score += -100 if de > 200 else (100 if de < 100 else 0)

    pm = fundamentals["profitMargin"]
    if pm is not None:
        f_count += 1
        f_score += 100 if pm > 0.2 else (50 if pm > 0 else -100)

    roe = fundamentals["returnOnEquity"]
    if roe is not None:
        f_count += 1
        f_score += 100 if roe > 0.15 else (50 if roe > 0 else -100)

    fundamental_score = round(f_score / f_count) if f_count > 0 else 0

    # ── LAYER 2: TECHNICALS ────────────────────
    hist = stock.history(period="1y")
    if hist.empty:
        return {"error": "No historical price data available."}

    closes = hist["Close"]
    volumes = hist["Volume"]

    # Moving Averages
    sma50 = closes.rolling(window=50).mean()
    sma200 = closes.rolling(window=200).mean()

    # RSI (14-period)
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Bollinger Bands (20, 2)
    bb_mid = closes.rolling(window=20).mean()
    bb_std = closes.rolling(window=20).std()
    bb_upper = bb_mid + (bb_std * 2)
    bb_lower = bb_mid - (bb_std * 2)

    latest_close = float(closes.iloc[-1])
    latest_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None
    latest_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None
    latest_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
    latest_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
    latest_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None

    # Score technicals: -100 to +100
    t_score, t_count = 0, 0

    if latest_sma200:
        t_count += 1
        t_score += 100 if latest_close > latest_sma200 else -100

    if latest_rsi:
        t_count += 1
        if latest_rsi > 70:
            t_score -= 100
        elif latest_rsi < 30:
            t_score += 100
        else:
            t_score += 0

    if latest_macd is not None and latest_signal is not None:
        t_count += 1
        t_score += 100 if latest_macd > latest_signal else -100

    # Golden/Death cross - FIXED SYNTAX (added colon)
    if latest_sma50 and latest_sma200:
        t_count += 1
        t_score += 100 if latest_sma50 > latest_sma200 else -100

    technical_score = round(t_score / t_count) if t_count > 0 else 0

    # Build chart dataframe (last 120 days for a cleaner chart)
    chart_len = min(120, len(closes))
    chart_df = pd.DataFrame({
        "Date": closes.index[-chart_len:],
        "Price": closes.values[-chart_len:],
        "SMA 50": sma50.values[-chart_len:],
        "SMA 200": sma200.values[-chart_len:],
        "Upper Band": bb_upper.values[-chart_len:],
        "Lower Band": bb_lower.values[-chart_len:],
    })

    volume_df = pd.DataFrame({
        "Date": volumes.index[-chart_len:],
        "Volume": volumes.values[-chart_len:],
    })

    technicals = {
        "price": latest_close,
        "sma50": latest_sma50,
        "sma200": latest_sma200,
        "rsi": latest_rsi,
        "macd": latest_macd,
        "macd_signal": latest_signal,
        "chart_df": chart_df,
        "volume_df": volume_df,
    }

    # ── LAYER 3: SENTIMENT (FinBERT) ───────────
    headlines = []
    news_links = []
    try:
        news_items = stock.news or []
        for item in news_items[:10]:
            title = ""
            link = ""

            # New yfinance format (0.2.40+): nested under "content"
            if isinstance(item, dict) and "content" in item:
                content = item["content"]
                title = content.get("title", "") if isinstance(content, dict) else ""
                link = content.get("canonicalUrl", {}).get("url", "") if isinstance(content, dict) else ""

            # Old yfinance format: flat dict with "title" and "link"
            elif isinstance(item, dict):
                title = item.get("title", "")
                link = item.get("link", "")

            # Some versions return objects with attributes
            elif hasattr(item, "title"):
                title = getattr(item, "title", "")
                link = getattr(item, "link", "")

            if title:
                headlines.append(title)
                news_links.append(link)
    except Exception as e:
        st.warning(f"Could not fetch news: {e}")

    if DEBUG_MODE:
        print(f"\n📰 [NEWS DEBUG] {ticker}: Found {len(headlines)} headlines")
        if headlines:
            print(f"  First headline: {headlines[0][:100]}...")
        else:
            print(f"  No headlines found. Raw news type: {type(stock.news)}")
            if stock.news:
                print(f"  First raw news item: {str(stock.news[0])[:200]}")

    sentiment_results = analyze_sentiment(headlines) if headlines else []

    # Add links back into results
    for i, result in enumerate(sentiment_results):
        if i < len(news_links):
            result["link"] = news_links[i]

    if sentiment_results:
        sentiment_score = round(sum(r["score"] for r in sentiment_results) / len(sentiment_results))
    else:
        sentiment_score = 0

    if DEBUG_MODE:
        print(f"\n📊 [SCORES] Fundamental: {fundamental_score}, Technical: {technical_score}, Sentiment: {sentiment_score}")

    # ── COMPOSITE SCORE ────────────────────────
    composite = round(
        (fundamental_score * 0.4) +
        (technical_score * 0.3) +
        (sentiment_score * 0.3)
    )

    # ── DUAL RECOMMENDATION SYSTEM ─────────────

    # === SIGNAL 1: "If You Own It" (Quality + Sell Triggers) ===
    # Focuses on: are fundamentals deteriorating? Any reason to exit?
    owner_score = 0
    owner_reasons = []

    # Strong fundamentals = keep holding
    if fundamental_score > 20:
        owner_score += 2
        owner_reasons.append("Fundamentals remain solid")
    elif fundamental_score < -30:
        owner_score -= 2
        owner_reasons.append("Fundamentals are deteriorating")

    # No bearish technical breakdown = keep holding
    if latest_sma200 and latest_close > latest_sma200:
        owner_score += 1
        owner_reasons.append("Price above 200-day SMA support")
    elif latest_sma200 and latest_close < latest_sma200:
        owner_score -= 1
        owner_reasons.append("Price has broken below 200-day SMA")

    # RSI not extremely overbought = no urgent sell
    if latest_rsi and latest_rsi > 80:
        owner_score -= 1
        owner_reasons.append("RSI extremely overbought — consider trimming")
    elif latest_rsi and latest_rsi < 25:
        owner_score += 1
        owner_reasons.append("RSI deeply oversold — potential bounce ahead")

    # Sentiment not collapsing
    if sentiment_score < -40:
        owner_score -= 1
        owner_reasons.append("Negative news sentiment building")
    elif sentiment_score > 20:
        owner_score += 1
        owner_reasons.append("Positive news sentiment")

    # Death cross = warning
    if latest_sma50 and latest_sma200 and latest_sma50 < latest_sma200:
        owner_score -= 1
        owner_reasons.append("Death cross detected (50-day below 200-day)")

    if owner_score >= 2:
        owner_rec = "HOLD"
        owner_summary = "No sell triggers detected. Fundamentals support continued holding."
    elif owner_score >= 0:
        owner_rec = "HOLD"
        owner_summary = "Mixed signals, but no strong reason to exit the position."
    elif owner_score >= -2:
        owner_rec = "CAUTION"
        owner_summary = "Some warning signs emerging. Consider reviewing your position."
    else:
        owner_rec = "SELL"
        owner_summary = "Multiple sell triggers active. Consider reducing exposure."

    # === SIGNAL 2: "If You're Considering Buying" (Quality + Entry Timing) ===
    # Focuses on: is this a GOOD ENTRY POINT right now?
    buyer_score = 0
    buyer_reasons = []

    # Quality foundation — good company?
    if fundamental_score > 30:
        buyer_score += 2
        buyer_reasons.append("Strong fundamental quality")
    elif fundamental_score > 0:
        buyer_score += 1
        buyer_reasons.append("Decent fundamental profile")
    elif fundamental_score < -30:
        buyer_score -= 2
        buyer_reasons.append("Weak fundamentals — higher risk")

    # Entry timing — where in the 52-week range?
    high_52 = fundamentals.get("fiftyTwoWeekHigh")
    low_52 = fundamentals.get("fiftyTwoWeekLow")
    if high_52 and low_52 and high_52 != low_52:
        range_pct = (latest_close - low_52) / (high_52 - low_52)
        if range_pct < 0.3:
            buyer_score += 2
            buyer_reasons.append(f"Trading in bottom 30% of 52-week range — attractive entry")
        elif range_pct < 0.5:
            buyer_score += 1
            buyer_reasons.append(f"Below midpoint of 52-week range")
        elif range_pct > 0.9:
            buyer_score -= 2
            buyer_reasons.append(f"Near 52-week highs — limited upside, elevated risk")
        elif range_pct > 0.75:
            buyer_score -= 1
            buyer_reasons.append(f"In upper quartile of 52-week range")

    # RSI — oversold = good entry, overbought = bad entry
    if latest_rsi and latest_rsi < 35:
        buyer_score += 2
        buyer_reasons.append(f"RSI at {latest_rsi:.0f} — oversold, potential bounce")
    elif latest_rsi and latest_rsi < 45:
        buyer_score += 1
        buyer_reasons.append(f"RSI at {latest_rsi:.0f} — pulled back from highs")
    elif latest_rsi and latest_rsi > 70:
        buyer_score -= 2
        buyer_reasons.append(f"RSI at {latest_rsi:.0f} — overbought, wait for pullback")

    # Price near SMA200 support = good entry
    if latest_sma200:
        dist_from_sma200 = (latest_close - latest_sma200) / latest_sma200
        if -0.03 <= dist_from_sma200 <= 0.03:
            buyer_score += 1
            buyer_reasons.append("Price near 200-day SMA — potential support level")
        elif dist_from_sma200 < -0.05:
            buyer_score += 1
            buyer_reasons.append("Price well below 200-day SMA — deep value territory")

    # Price near lower Bollinger Band = good entry
    latest_bb_lower = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None
    if latest_bb_lower:
        if latest_close <= latest_bb_lower * 1.02:
            buyer_score += 1
            buyer_reasons.append("Price near lower Bollinger Band — at bottom of volatility range")

    # Forward P/E vs Trailing P/E — earnings growth expected?
    fpe = fundamentals.get("forwardPE")
    tpe = fundamentals.get("peRatio")
    if fpe and tpe and fpe < tpe * 0.85:
        buyer_score += 1
        buyer_reasons.append("Forward P/E significantly below trailing — earnings growth expected")

    # Positive sentiment = tailwind for new position
    if sentiment_score > 30:
        buyer_score += 1
        buyer_reasons.append("Strong positive news sentiment")
    elif sentiment_score < -30:
        buyer_score -= 1
        buyer_reasons.append("Negative sentiment — headline risk")

    if buyer_score >= 4:
        buyer_rec = "STRONG BUY"
        buyer_summary = "Excellent entry point. Strong quality with favorable timing."
    elif buyer_score >= 2:
        buyer_rec = "BUY"
        buyer_summary = "Good entry opportunity. Quality and timing align."
    elif buyer_score >= 0:
        buyer_rec = "WAIT"
        buyer_summary = "Decent company, but entry timing could be better. Consider waiting for a pullback."
    elif buyer_score >= -2:
        buyer_rec = "AVOID"
        buyer_summary = "Unfavorable entry conditions. Wait for a better setup."
    else:
        buyer_rec = "STRONG AVOID"
        buyer_summary = "Poor quality and/or terrible timing. Stay away for now."

    confidence = min(abs(composite), 100)

    if DEBUG_MODE:
        print(f"\n✅ [RECOMMENDATIONS] Owner: {owner_rec} (score: {owner_score}), Buyer: {buyer_rec} (score: {buyer_score})")
        print(f"{'='*60}\n")

    return {
        "ticker": ticker,
        "name": fundamentals.get("shortName", ticker),
        "sector": fundamentals.get("sector", "N/A"),
        "industry": fundamentals.get("industry", "N/A"),
        "owner_rec": owner_rec,
        "owner_summary": owner_summary,
        "owner_reasons": owner_reasons,
        "buyer_rec": buyer_rec,
        "buyer_summary": buyer_summary,
        "buyer_reasons": buyer_reasons,
        "confidence": confidence,
        "composite_score": composite,
        "fundamental_score": fundamental_score,
        "technical_score": technical_score,
        "sentiment_score": sentiment_score,
        "fundamentals": fundamentals,
        "technicals": technicals,
        "sentiment_results": sentiment_results,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_price_chart(chart_df: pd.DataFrame, volume_df: pd.DataFrame, ticker: str):
    """Build a Plotly chart with price, SMAs, Bollinger Bands, and volume."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    # Bollinger Bands fill
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["Upper Band"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["Lower Band"],
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Bollinger Band lines
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["Upper Band"],
        line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
        name="Bollinger Bands", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["Lower Band"],
        line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"),
        showlegend=False,
    ), row=1, col=1)

    # SMA 200
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["SMA 200"],
        line=dict(color="#f59e0b", width=1.5),
        name="SMA 200",
    ), row=1, col=1)

    # SMA 50
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["SMA 50"],
        line=dict(color="#a855f7", width=1.5),
        name="SMA 50",
    ), row=1, col=1)

    # Price
    fig.add_trace(go.Scatter(
        x=chart_df["Date"], y=chart_df["Price"],
        line=dict(color="#3b82f6", width=2.5),
        name="Price",
    ), row=1, col=1)

    # Volume bars
    colors = ["#2ea043" if chart_df["Price"].iloc[i] >= chart_df["Price"].iloc[max(0, i-1)]
              else "#f85149" for i in range(len(chart_df))]
    fig.add_trace(go.Bar(
        x=volume_df["Date"], y=volume_df["Volume"],
        marker_color=colors, marker_opacity=0.4,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#8b949e"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False, row=1, col=1,
                     tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False, row=2, col=1)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>AlphaScope AI</h1>
        <p style="color: #8b949e; font-size: 1.05rem; max-width: 600px; margin: 0.5rem auto 0 auto; line-height: 1.6;">
            Hi! I'm your AI-powered stock analyst. Give me any ticker and I'll run a 
            <span style="color: #58a6ff;">three-layer deep dive</span> — fundamentals from Yahoo Finance, 
            technical indicators computed in real-time, and NLP sentiment analysis on recent news 
            using <span style="color: #a78bfa;">FinBERT</span>. I'll score everything, give you a 
            Buy/Hold/Sell recommendation, and write you a full analyst briefing.
            I'll even give you <span style="color: #2ea043;">separate signals</span> for whether you already own the stock 
            or you're thinking about buying in.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search bar with autocomplete
    col_pad1, col_search, col_pad2 = st.columns([1, 3, 1])
    with col_search:
        ticker_input_raw = st.text_input(
            "search_bar",
            placeholder="Search by ticker or company name (e.g. AAPL, Tesla, NVDA)...",
            label_visibility="collapsed",
            key="ticker_search",
        )

        # Show autocomplete suggestions as the user types
        if ticker_input_raw and not any(ticker_input_raw.upper() == t for t in POPULAR_TICKERS):
            suggestions = search_tickers(ticker_input_raw)
            if suggestions:
                suggestion_html = ""
                for s in suggestions:
                    t = s.split(" — ")[0]
                    n = s.split(" — ")[1] if " — " in s else ""
                    suggestion_html += f'<div style="padding: 0.5rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.85rem;"><span style="color: #58a6ff; font-weight: 600;">{t}</span> <span style="color: #8b949e;">— {n}</span></div>'
                st.markdown(f"""
                <div style="background: #161b22; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; margin-top: -0.5rem; overflow: hidden;">
                    {suggestion_html}
                </div>
                <p style="text-align: center; color: rgba(139,148,158,0.4); font-size: 0.7rem; margin-top: 0.4rem;">
                    Select a suggestion by typing the full ticker, then press Enter
                </p>
                """, unsafe_allow_html=True)

    # Resolve ticker from input — check if user typed a company name
    ticker_input = ""
    if ticker_input_raw:
        raw = ticker_input_raw.strip().upper()
        # Direct ticker match
        if raw in POPULAR_TICKERS:
            ticker_input = raw
        else:
            # Check if they typed a company name
            for t, name in POPULAR_TICKERS.items():
                if raw == t or ticker_input_raw.strip().lower() == name.lower():
                    ticker_input = t
                    break
            # If no match in our list, assume it's a valid ticker anyway
            if not ticker_input:
                ticker_input = raw

    if not ticker_input:
        # Empty state — feature cards
        st.markdown("""
        <div style="max-width: 700px; margin: 2rem auto 0 auto;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 2rem;">
                <div class="score-card" style="text-align: left;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                    <h3 style="text-align: left; margin-bottom: 0.3rem;">Fundamentals</h3>
                    <p style="color: #8b949e; font-size: 0.78rem; line-height: 1.5; margin: 0;">
                        P/E, revenue growth, margins, debt ratios — scored and graded automatically
                    </p>
                </div>
                <div class="score-card" style="text-align: left;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📈</div>
                    <h3 style="text-align: left; margin-bottom: 0.3rem;">Technicals</h3>
                    <p style="color: #8b949e; font-size: 0.78rem; line-height: 1.5; margin: 0;">
                        SMA, RSI, MACD, Bollinger Bands — computed from 1 year of price data
                    </p>
                </div>
                <div class="score-card" style="text-align: left;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                    <h3 style="text-align: left; margin-bottom: 0.3rem;">NLP Sentiment</h3>
                    <p style="color: #8b949e; font-size: 0.78rem; line-height: 1.5; margin: 0;">
                        FinBERT classifies news headlines as positive, negative, or neutral locally
                    </p>
                </div>
            </div>
            <div style="text-align: center; margin-top: 2.5rem;">
                <p style="color: rgba(139,148,158,0.4); font-size: 0.78rem;">
                    Try searching for <span style="color: #58a6ff;">AAPL</span>, 
                    <span style="color: #58a6ff;">TSLA</span>, 
                    <span style="color: #58a6ff;">NVDA</span>, or 
                    <span style="color: #58a6ff;">MSFT</span> to get started
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    ticker = ticker_input

    # Run analysis
    with st.spinner(f"Analyzing {ticker} — pulling fundamentals, computing technicals, running FinBERT sentiment..."):
        data = run_analysis(ticker)

    if "error" in data:
        st.error(data["error"])
        return

    # ── TOP ROW: Dual Recommendations + Score Cards ──
    rec_col1, rec_col2, score_col = st.columns([1, 1, 2])

    with rec_col1:
        owner_class = data["owner_rec"].lower().replace(" ", "-")
        reasons_html = "".join(f'<div class="rec-reason">• {r}</div>' for r in data["owner_reasons"][:3])
        st.markdown(f"""
        <div class="rec-badge {owner_class}" style="padding: 1.2rem;">
            <div class="rec-label">📌 If You Own It</div>
            <h2>{data['owner_rec']}</h2>
            <div class="rec-summary">{data['owner_summary']}</div>
            <div style="margin-top: 0.5rem;">{reasons_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        buyer_class = data["buyer_rec"].lower().replace(" ", "-")
        reasons_html = "".join(f'<div class="rec-reason">• {r}</div>' for r in data["buyer_reasons"][:3])
        st.markdown(f"""
        <div class="rec-badge {buyer_class}" style="padding: 1.2rem;">
            <div class="rec-label">🛒 If You're Looking to Buy</div>
            <h2>{data['buyer_rec']}</h2>
            <div class="rec-summary">{data['buyer_summary']}</div>
            <div style="margin-top: 0.5rem;">{reasons_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with score_col:
        s1, s2, s3 = st.columns(3)
        for col, title, score in [
            (s1, "Fundamentals", data["fundamental_score"]),
            (s2, "Technicals", data["technical_score"]),
            (s3, "Sentiment", data["sentiment_score"]),
        ]:
            with col:
                css_class = score_color_class(score)
                st.markdown(f"""
                <div class="score-card">
                    <h3>{title}</h3>
                    <div class="score-value {css_class}">{score_prefix(score)}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI NARRATIVE ──
    with st.spinner("Generating AI narrative..."):
        recommendation_summary = f"Owner Signal: {data['owner_rec']} | Buyer Signal: {data['buyer_rec']}"
        narrative = generate_narrative(
            ticker,
            data["fundamentals"],
            data["technicals"],
            data["sentiment_score"],
            data["composite_score"],
            recommendation_summary,
        )

    # Clean up narrative: convert markdown to HTML
    import re
    clean_narrative = narrative
    # Handle newlines
    clean_narrative = clean_narrative.replace(chr(10)+chr(10), '<br><br>')
    clean_narrative = clean_narrative.replace(chr(10), '<br>')
    # ***bold italic*** → <strong><em>
    clean_narrative = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', clean_narrative)
    # **bold** → <strong>
    clean_narrative = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', clean_narrative)
    # *italic* → <em>
    clean_narrative = re.sub(r'\*(.+?)\*', r'<em>\1</em>', clean_narrative)
    # __bold__ → <strong>
    clean_narrative = re.sub(r'__(.+?)__', r'<strong>\1</strong>', clean_narrative)
    # _italic_ → <em>
    clean_narrative = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<em>\1</em>', clean_narrative)
    # Markdown headers (## Header) → bold text
    clean_narrative = re.sub(r'#{1,4}\s*(.+?)(<br>|$)', r'<strong>\1</strong><br>', clean_narrative)
    # Bullet points
    clean_narrative = re.sub(r'<br>[\-\•]\s*', '<br>• ', clean_narrative)

    st.markdown(f"""
    <div class="glass-panel">
        <h3>🤖 AI Executive Summary — {data['name']} ({ticker})</h3>
        <div style="color: rgba(230,237,243,0.8); line-height: 1.7; font-size: 0.95rem;">
            {clean_narrative}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHART + FUNDAMENTALS ROW ──
    chart_col, fund_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="glass-panel"><h3>📈 Technical Price Action</h3></div>', unsafe_allow_html=True)
        fig = build_price_chart(
            data["technicals"]["chart_df"],
            data["technicals"]["volume_df"],
            ticker,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with fund_col:
        fund = data["fundamentals"]
        metrics = [
            ("P/E Ratio", fmt_ratio(fund.get("peRatio"))),
            ("Forward P/E", fmt_ratio(fund.get("forwardPE"))),
            ("Market Cap", fmt_large_number(fund.get("marketCap"))),
            ("EPS", fmt_price(fund.get("eps"))),
            ("Revenue Growth", fmt_pct(fund.get("revenueGrowth"))),
            ("Profit Margin", fmt_pct(fund.get("profitMargin"))),
            ("Debt/Equity", fmt_ratio(fund.get("debtToEquity"))),
            ("ROE", fmt_pct(fund.get("returnOnEquity"))),
            ("Dividend Yield", fmt_pct(fund.get("dividendYield"))),
            ("52W High", fmt_price(fund.get("fiftyTwoWeekHigh"))),
            ("52W Low", fmt_price(fund.get("fiftyTwoWeekLow"))),
            ("Free Cash Flow", fmt_large_number(fund.get("freeCashflow"))),
        ]
        rows_html = "".join(
            f'<div class="metric-row"><span class="metric-label">{label}</span><span class="metric-value">{value}</span></div>'
            for label, value in metrics if value != "N/A"
        )
        st.markdown(f"""
        <div class="glass-panel">
            <h3>📊 Fundamentals</h3>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # ── TECHNICAL INDICATORS SUMMARY ──
    tech = data["technicals"]
    t1, t2, t3, t4 = st.columns(4)
    indicators = [
        (t1, "RSI (14)", f"{tech['rsi']:.1f}" if tech["rsi"] else "N/A",
         "Overbought" if tech["rsi"] and tech["rsi"] > 70 else "Oversold" if tech["rsi"] and tech["rsi"] < 30 else "Neutral"),
        (t2, "MACD", f"{tech['macd']:.3f}" if tech["macd"] else "N/A",
         "Bullish" if tech["macd"] and tech["macd_signal"] and tech["macd"] > tech["macd_signal"] else "Bearish"),
        (t3, "SMA 50", fmt_price(tech["sma50"]),
         "Above price" if tech["sma50"] and tech["sma50"] > tech["price"] else "Below price"),
        (t4, "SMA 200", fmt_price(tech["sma200"]),
         "Above price" if tech["sma200"] and tech["sma200"] > tech["price"] else "Below price"),
    ]
    for col, name, value, signal in indicators:
        with col:
            sig_color = "#2ea043" if "Bullish" in signal or signal == "Oversold" or signal == "Below price" else "#f85149" if "Bearish" in signal or signal == "Overbought" or signal == "Above price" else "#d29922"
            st.markdown(f"""
            <div class="score-card">
                <h3>{name}</h3>
                <div class="score-value" style="font-size:1.5rem; color: #e6edf3;">{value}</div>
                <div style="font-size: 0.75rem; color: {sig_color}; margin-top: 0.3rem; font-weight: 600;">{signal}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── NEWS + CHAT ROW ──
    news_col, chat_col = st.columns([1, 2])

    with news_col:
        sr = data["sentiment_results"]
        if sr:
            from html import escape
            news_html = ""
            for item in sr:
                sent = item["sentiment"]
                css = f"sentiment-{sent}"
                safe_headline = escape(item['headline'])
                conf = item['confidence']
                news_html += (
                    f'<div class="news-item">'
                    f'<div class="news-headline">{safe_headline}</div>'
                    f'<div class="news-meta">'
                    f'<span class="news-source">FinBERT: {conf:.0%}</span>'
                    f'<span class="sentiment-tag {css}">{sent}</span>'
                    f'</div></div>'
                )
            st.markdown(
                f'<div class="glass-panel">'
                f'<h3>📰 News Sentiment (FinBERT)</h3>'
                f'{news_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="glass-panel">
                <h3>📰 News Sentiment</h3>
                <p style="color: #8b949e                <p style="color: #8b949e;">No recent news available for this ticker.</p>
            </div>
            """, unsafe_allow_html=True)

    with chat_col:
        st.markdown("""
        <div class="glass-panel">
            <h3>💬 AlphaScope Assistant</h3>
            <p style="color: #8b949e; font-size: 0.85rem; margin-bottom: 0.5rem;">
                Ask follow-up questions about this analysis — grounded in the data above.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize chat history for this ticker
        if "chat_ticker" not in st.session_state or st.session_state.chat_ticker != ticker:
            st.session_state.chat_ticker = ticker
            st.session_state.chat_history = []

        # Build context for RAG (exclude non-serializable chart dataframes)
        chat_context = {
            "ticker": ticker,
            "owner_recommendation": data["owner_rec"],
            "owner_summary": data["owner_summary"],
            "owner_reasons": data["owner_reasons"],
            "buyer_recommendation": data["buyer_rec"],
            "buyer_summary": data["buyer_summary"],
            "buyer_reasons": data["buyer_reasons"],
            "confidence": data["confidence"],
            "composite_score": data["composite_score"],
            "fundamental_score": data["fundamental_score"],
            "technical_score": data["technical_score"],
            "sentiment_score": data["sentiment_score"],
            "fundamentals": {k: v for k, v in data["fundamentals"].items() if v is not None},
            "technicals": {
                "price": tech["price"],
                "sma50": tech["sma50"],
                "sma200": tech["sma200"],
                "rsi": tech["rsi"],
                "macd": tech["macd"],
            },
            "sentiment_headlines": [
                {"headline": r["headline"], "sentiment": r["sentiment"], "score": r["score"]}
                for r in data["sentiment_results"]
            ],
        }

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask about {ticker}... (e.g. 'What's the biggest risk?')"):
            # Show user message
            with st.chat_message("user"):
                st.write(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_groq(ticker, prompt, chat_context)
                st.write(response)

            # Save to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="disclaimer">
        This analysis is generated algorithmically for educational purposes only.
        It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
        <br><br>
        Built with Streamlit · yfinance · HuggingFace FinBERT · Groq Llama 3.3
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()