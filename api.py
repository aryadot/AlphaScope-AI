"""
api.py — AlphaScope AI REST API
================================
FastAPI backend exposing the AlphaScope agentic investment research
engine as a structured REST API with Pydantic request validation.

Endpoints:
    GET  /health          — Health check
    POST /recommend       — Full agentic recommendation for a ticker
    POST /fundamentals    — Fundamental analysis only
    POST /technicals      — Technical analysis only
    POST /sentiment       — Sentiment analysis only

Usage:
    uvicorn api:app --reload
"""

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from agent import run_agentic_analysis, get_fundamentals, get_technicals, run_sentiment

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaScope AI API",
    description="REST API exposing the AlphaScope agentic investment research engine",
    version="1.0.0",
)


# ── Request / Response Schemas ────────────────────────────────────────────────
class TickerRequest(BaseModel):
    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Ticker cannot be empty")
        if len(v) > 10:
            raise ValueError("Ticker symbol too long — maximum 10 characters")
        if not v.isalpha():
            raise ValueError("Ticker must contain letters only")
        return v


class ScoreBreakdown(BaseModel):
    fundamental_score: int | None = None
    technical_score:   int | None = None
    sentiment_score:   int | None = None
    composite_score:   float | None = None


class RecommendationResponse(BaseModel):
    ticker:         str
    recommendation: str          # BUY | HOLD | SELL
    confidence:     str          # HIGH | MODERATE | LOW
    analysis:       str
    scores:         ScoreBreakdown


class AnalysisResponse(BaseModel):
    ticker: str
    data:   dict


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set")
    return key


def _parse_tool_result(raw: str) -> dict:
    """Safely parse a tool result string to dict."""
    try:
        if raw.startswith("{"):
            return json.loads(raw)
    except Exception:
        pass
    return {}


def _derive_recommendation(composite: float) -> tuple[str, str]:
    """Map composite score to recommendation and confidence label."""
    if composite > 30:
        return "BUY",  "HIGH" if composite > 60 else "MODERATE"
    if composite < -30:
        return "SELL", "HIGH" if composite < -60 else "MODERATE"
    return "HOLD", "MODERATE"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health_check():
    """Returns service status. Used by CI/CD and monitoring."""
    return {"status": "ok", "service": "AlphaScope AI API", "version": "1.0.0"}


@app.post("/recommend", response_model=RecommendationResponse, summary="Full agentic recommendation")
def get_recommendation(request: TickerRequest):
    """
    Runs the full AlphaScope agentic pipeline for a ticker:
    1. Fetches fundamental metrics via yfinance
    2. Computes technical indicators (SMA, RSI, MACD)
    3. Runs FinBERT sentiment on recent news headlines
    4. LLaMA 3.3 synthesizes a structured investment report
    Returns a structured recommendation with confidence and score breakdown.
    """
    groq_key = _require_groq_key()

    try:
        fund_data = _parse_tool_result(get_fundamentals.invoke(request.ticker))
        tech_data = _parse_tool_result(get_technicals.invoke(request.ticker))
        sent_data = _parse_tool_result(run_sentiment.invoke(request.ticker))

        fund_score = fund_data.get("fundamental_score", 0) or 0
        tech_score = tech_data.get("technical_score",   0) or 0
        sent_score = sent_data.get("aggregate_sentiment_score", 0) or 0
        composite  = round((fund_score + tech_score + sent_score) / 3, 2)

        recommendation, confidence = _derive_recommendation(composite)
        analysis = run_agentic_analysis(request.ticker, groq_key)

        return RecommendationResponse(
            ticker=request.ticker,
            recommendation=recommendation,
            confidence=confidence,
            analysis=analysis,
            scores=ScoreBreakdown(
                fundamental_score=fund_score,
                technical_score=tech_score,
                sentiment_score=sent_score,
                composite_score=composite,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/fundamentals", response_model=AnalysisResponse, summary="Fundamental analysis only")
def get_fundamental_analysis(request: TickerRequest):
    """Returns fundamental financial metrics and scoring for a ticker."""
    try:
        data = _parse_tool_result(get_fundamentals.invoke(request.ticker))
        return AnalysisResponse(ticker=request.ticker, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/technicals", response_model=AnalysisResponse, summary="Technical analysis only")
def get_technical_analysis(request: TickerRequest):
    """Returns technical indicators (SMA, RSI, MACD) and scoring for a ticker."""
    try:
        data = _parse_tool_result(get_technicals.invoke(request.ticker))
        return AnalysisResponse(ticker=request.ticker, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment", response_model=AnalysisResponse, summary="Sentiment analysis only")
def get_sentiment_analysis(request: TickerRequest):
    """Runs FinBERT sentiment on recent news headlines for a ticker."""
    try:
        data = _parse_tool_result(run_sentiment.invoke(request.ticker))
        return AnalysisResponse(ticker=request.ticker, data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
