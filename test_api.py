"""
tests/test_api.py — Unit Tests for AlphaScope AI API
======================================================
Tests cover:
  - Health endpoint response
  - Input validation (empty ticker, invalid characters, too long)
  - Response schema structure
  - Score-to-recommendation mapping logic
  - API behavior with mocked external dependencies

Run with:
    pytest tests/ -v
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# ── Mock heavy dependencies before importing api ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the entire agent module so tests run without langchain/yfinance installed
mock_agent = MagicMock()
mock_agent.get_fundamentals.invoke.return_value = json.dumps({"fundamental_score": 50})
mock_agent.get_technicals.invoke.return_value   = json.dumps({"technical_score": 40})
mock_agent.run_sentiment.invoke.return_value    = json.dumps({"aggregate_sentiment_score": 30})
mock_agent.run_agentic_analysis.return_value    = "Mock analysis complete."
sys.modules["agent"] = mock_agent

from api import app, _derive_recommendation, _parse_tool_result
from fastapi.testclient import TestClient

client = TestClient(app)


# ── Health Check ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_returns_service_name(self):
        response = client.get("/health")
        assert "AlphaScope" in response.json()["service"]

    def test_health_returns_version(self):
        response = client.get("/health")
        assert "version" in response.json()


# ── Input Validation ──────────────────────────────────────────────────────────
class TestTickerValidation:
    def test_empty_ticker_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": ""})
        assert response.status_code == 422

    def test_whitespace_ticker_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "   "})
        assert response.status_code == 422

    def test_ticker_too_long_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "TOOLONGTICKER"})
        assert response.status_code == 422

    def test_ticker_with_numbers_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "AAPL123"})
        assert response.status_code == 422

    def test_ticker_normalized_to_uppercase(self):
        response = client.post("/fundamentals", json={"ticker": "aapl"})
        assert response.status_code == 200
        assert response.json()["ticker"] == "AAPL"

    def test_missing_ticker_field_returns_422(self):
        response = client.post("/fundamentals", json={})
        assert response.status_code == 422


# ── Recommendation Logic ──────────────────────────────────────────────────────
class TestRecommendationLogic:
    def test_high_composite_returns_buy_high(self):
        rec, conf = _derive_recommendation(70)
        assert rec == "BUY" and conf == "HIGH"

    def test_moderate_positive_returns_buy_moderate(self):
        rec, conf = _derive_recommendation(40)
        assert rec == "BUY" and conf == "MODERATE"

    def test_neutral_returns_hold(self):
        rec, _ = _derive_recommendation(0)
        assert rec == "HOLD"

    def test_low_positive_returns_hold(self):
        rec, _ = _derive_recommendation(20)
        assert rec == "HOLD"

    def test_low_negative_returns_hold(self):
        rec, _ = _derive_recommendation(-20)
        assert rec == "HOLD"

    def test_moderate_negative_returns_sell_moderate(self):
        rec, conf = _derive_recommendation(-40)
        assert rec == "SELL" and conf == "MODERATE"

    def test_high_negative_returns_sell_high(self):
        rec, conf = _derive_recommendation(-70)
        assert rec == "SELL" and conf == "HIGH"

    def test_boundary_exactly_30_is_hold(self):
        rec, _ = _derive_recommendation(30)
        assert rec == "HOLD"

    def test_boundary_just_above_30_is_buy(self):
        rec, _ = _derive_recommendation(31)
        assert rec == "BUY"


# ── Tool Result Parser ────────────────────────────────────────────────────────
class TestParseToolResult:
    def test_valid_json_returns_dict(self):
        result = _parse_tool_result('{"ticker": "AAPL", "score": 50}')
        assert result == {"ticker": "AAPL", "score": 50}

    def test_error_string_returns_empty_dict(self):
        result = _parse_tool_result("Error fetching data for INVALID")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _parse_tool_result("")
        assert result == {}

    def test_non_json_string_returns_empty_dict(self):
        result = _parse_tool_result("No data found")
        assert result == {}


# ── Endpoint Structure ────────────────────────────────────────────────────────
class TestFundamentalsEndpoint:
    def test_returns_200(self):
        response = client.post("/fundamentals", json={"ticker": "AAPL"})
        assert response.status_code == 200

    def test_returns_ticker(self):
        response = client.post("/fundamentals", json={"ticker": "AAPL"})
        assert response.json()["ticker"] == "AAPL"

    def test_returns_data_field(self):
        response = client.post("/fundamentals", json={"ticker": "AAPL"})
        assert "data" in response.json()


class TestTechnicalsEndpoint:
    def test_returns_200(self):
        response = client.post("/technicals", json={"ticker": "MSFT"})
        assert response.status_code == 200

    def test_returns_ticker(self):
        response = client.post("/technicals", json={"ticker": "MSFT"})
        assert response.json()["ticker"] == "MSFT"


class TestSentimentEndpoint:
    def test_returns_200(self):
        response = client.post("/sentiment", json={"ticker": "GOOGL"})
        assert response.status_code == 200

    def test_returns_data_field(self):
        response = client.post("/sentiment", json={"ticker": "GOOGL"})
        assert "data" in response.json()


class TestRecommendEndpoint:
    def test_returns_valid_recommendation(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert response.status_code == 200
            assert response.json()["recommendation"] in ["BUY", "HOLD", "SELL"]

    def test_returns_confidence_field(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert "confidence" in response.json()

    def test_returns_scores_breakdown(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
            response = client.post("/recommend", json={"ticker": "AAPL"})
            data = response.json()
            assert "scores" in data
            assert "composite_score" in data["scores"]

    def test_missing_groq_key_returns_500(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert response.status_code == 500



# ── Health Check ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self):
        response = client.get("/health")
        data = response.json()
        assert "AlphaScope" in data["service"]

    def test_health_returns_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data


# ── Input Validation ──────────────────────────────────────────────────────────
class TestTickerValidation:
    def test_empty_ticker_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": ""})
        assert response.status_code == 422

    def test_whitespace_ticker_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "   "})
        assert response.status_code == 422

    def test_ticker_too_long_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "TOOLONGTICKER"})
        assert response.status_code == 422

    def test_ticker_with_numbers_returns_422(self):
        response = client.post("/fundamentals", json={"ticker": "AAPL123"})
        assert response.status_code == 422

    def test_ticker_normalized_to_uppercase(self):
        """Lowercase ticker should be accepted and normalized."""
        with patch("api.get_fundamentals") as mock_fund:
            mock_fund.invoke.return_value = json.dumps({
                "ticker": "AAPL", "fundamental_score": 50
            })
            response = client.post("/fundamentals", json={"ticker": "aapl"})
            assert response.status_code == 200
            assert response.json()["ticker"] == "AAPL"

    def test_missing_ticker_field_returns_422(self):
        response = client.post("/fundamentals", json={})
        assert response.status_code == 422


# ── Recommendation Logic ──────────────────────────────────────────────────────
class TestRecommendationLogic:
    def test_high_composite_returns_buy(self):
        rec, conf = _derive_recommendation(70)
        assert rec == "BUY"
        assert conf == "HIGH"

    def test_moderate_positive_returns_buy_moderate(self):
        rec, conf = _derive_recommendation(40)
        assert rec == "BUY"
        assert conf == "MODERATE"

    def test_neutral_returns_hold(self):
        rec, conf = _derive_recommendation(0)
        assert rec == "HOLD"

    def test_low_positive_returns_hold(self):
        rec, conf = _derive_recommendation(20)
        assert rec == "HOLD"

    def test_low_negative_returns_hold(self):
        rec, conf = _derive_recommendation(-20)
        assert rec == "HOLD"

    def test_moderate_negative_returns_sell(self):
        rec, conf = _derive_recommendation(-40)
        assert rec == "SELL"
        assert conf == "MODERATE"

    def test_high_negative_returns_sell_high_confidence(self):
        rec, conf = _derive_recommendation(-70)
        assert rec == "SELL"
        assert conf == "HIGH"

    def test_boundary_exactly_30(self):
        rec, _ = _derive_recommendation(30)
        assert rec == "HOLD"

    def test_boundary_just_above_30(self):
        rec, _ = _derive_recommendation(31)
        assert rec == "BUY"


# ── Tool Result Parser ────────────────────────────────────────────────────────
class TestParseToolResult:
    def test_valid_json_returns_dict(self):
        result = _parse_tool_result('{"ticker": "AAPL", "score": 50}')
        assert result == {"ticker": "AAPL", "score": 50}

    def test_invalid_json_returns_empty_dict(self):
        result = _parse_tool_result("Error fetching data for INVALID")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _parse_tool_result("")
        assert result == {}

    def test_non_json_string_returns_empty_dict(self):
        result = _parse_tool_result("No data found")
        assert result == {}


# ── Endpoint Structure Tests ──────────────────────────────────────────────────
class TestFundamentalsEndpoint:
    def test_returns_ticker_in_response(self):
        with patch("api.get_fundamentals") as mock:
            mock.invoke.return_value = json.dumps({"fundamental_score": 60})
            response = client.post("/fundamentals", json={"ticker": "AAPL"})
            assert response.status_code == 200
            assert response.json()["ticker"] == "AAPL"

    def test_returns_data_field(self):
        with patch("api.get_fundamentals") as mock:
            mock.invoke.return_value = json.dumps({"fundamental_score": 60})
            response = client.post("/fundamentals", json={"ticker": "AAPL"})
            assert "data" in response.json()

    def test_tool_error_returns_500(self):
        with patch("api.get_fundamentals") as mock:
            mock.invoke.side_effect = Exception("yfinance connection failed")
            response = client.post("/fundamentals", json={"ticker": "AAPL"})
            assert response.status_code == 500


class TestTechnicalsEndpoint:
    def test_returns_200_with_valid_ticker(self):
        with patch("api.get_technicals") as mock:
            mock.invoke.return_value = json.dumps({"technical_score": 40})
            response = client.post("/technicals", json={"ticker": "MSFT"})
            assert response.status_code == 200

    def test_returns_ticker_in_response(self):
        with patch("api.get_technicals") as mock:
            mock.invoke.return_value = json.dumps({"technical_score": 40})
            response = client.post("/technicals", json={"ticker": "MSFT"})
            assert response.json()["ticker"] == "MSFT"


class TestSentimentEndpoint:
    def test_returns_200_with_valid_ticker(self):
        with patch("api.run_sentiment") as mock:
            mock.invoke.return_value = json.dumps({
                "aggregate_sentiment_score": 30,
                "overall_sentiment": "positive"
            })
            response = client.post("/sentiment", json={"ticker": "GOOGL"})
            assert response.status_code == 200

    def test_returns_data_field(self):
        with patch("api.run_sentiment") as mock:
            mock.invoke.return_value = json.dumps({"aggregate_sentiment_score": 30})
            response = client.post("/sentiment", json={"ticker": "GOOGL"})
            assert "data" in response.json()


class TestRecommendEndpoint:
    def test_returns_recommendation_field(self):
        with patch("api.get_fundamentals") as mf, \
             patch("api.get_technicals") as mt, \
             patch("api.run_sentiment") as ms, \
             patch("api.run_agentic_analysis") as ma, \
             patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):

            mf.invoke.return_value = json.dumps({"fundamental_score": 60})
            mt.invoke.return_value = json.dumps({"technical_score": 50})
            ms.invoke.return_value = json.dumps({"aggregate_sentiment_score": 40})
            ma.return_value = "AAPL shows strong fundamentals with bullish momentum."

            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert response.status_code == 200
            data = response.json()
            assert "recommendation" in data
            assert data["recommendation"] in ["BUY", "HOLD", "SELL"]

    def test_returns_confidence_field(self):
        with patch("api.get_fundamentals") as mf, \
             patch("api.get_technicals") as mt, \
             patch("api.run_sentiment") as ms, \
             patch("api.run_agentic_analysis") as ma, \
             patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):

            mf.invoke.return_value = json.dumps({"fundamental_score": 60})
            mt.invoke.return_value = json.dumps({"technical_score": 50})
            ms.invoke.return_value = json.dumps({"aggregate_sentiment_score": 40})
            ma.return_value = "Strong buy signal."

            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert "confidence" in response.json()

    def test_returns_scores_breakdown(self):
        with patch("api.get_fundamentals") as mf, \
             patch("api.get_technicals") as mt, \
             patch("api.run_sentiment") as ms, \
             patch("api.run_agentic_analysis") as ma, \
             patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):

            mf.invoke.return_value = json.dumps({"fundamental_score": 60})
            mt.invoke.return_value = json.dumps({"technical_score": 50})
            ms.invoke.return_value = json.dumps({"aggregate_sentiment_score": 40})
            ma.return_value = "Analysis complete."

            response = client.post("/recommend", json={"ticker": "AAPL"})
            data = response.json()
            assert "scores" in data
            assert "composite_score" in data["scores"]

    def test_missing_groq_key_returns_500(self):
        with patch.dict(os.environ, {}, clear=True):
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
            response = client.post("/recommend", json={"ticker": "AAPL"})
            assert response.status_code == 500
