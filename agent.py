"""
AlphaScope AI — LangChain Agentic Layer + ChromaDB RAG
=======================================================
Adds two capabilities to AlphaScope:
1. LangChain ReAct Agent with three tools that autonomously retrieves
   fundamentals, technicals, and sentiment — then synthesizes a report
2. ChromaDB vector store + RAG pipeline for grounded Q&A over
   news headlines and the generated narrative
"""

import os
import json
from typing import Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangChain Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Definitions
# Each tool is a function the LangChain agent
# can autonomously decide to call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_fundamentals(ticker: str) -> str:
    """
    Retrieves fundamental financial metrics for a stock ticker.
    Returns P/E ratio, revenue growth, profit margin, debt/equity,
    return on equity, market cap, EPS, and dividend yield.
    Use this first to understand the financial health of the company.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        if not info or not info.get("regularMarketPrice"):
            return f"No fundamental data found for {ticker}."

        fundamentals = {
            "ticker": ticker.upper(),
            "company_name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }

        # Score fundamentals
        score = 0
        count = 0

        if fundamentals["pe_ratio"]:
            count += 1
            score += 100 if fundamentals["pe_ratio"] < 15 else (0 if fundamentals["pe_ratio"] <= 25 else -100)

        if fundamentals["revenue_growth"] is not None:
            count += 1
            score += 100 if fundamentals["revenue_growth"] > 0.1 else (50 if fundamentals["revenue_growth"] > 0 else -100)

        if fundamentals["profit_margin"] is not None:
            count += 1
            score += 100 if fundamentals["profit_margin"] > 0.2 else (50 if fundamentals["profit_margin"] > 0 else -100)

        if fundamentals["debt_to_equity"] is not None:
            count += 1
            score += -100 if fundamentals["debt_to_equity"] > 200 else (100 if fundamentals["debt_to_equity"] < 100 else 0)

        if fundamentals["return_on_equity"] is not None:
            count += 1
            score += 100 if fundamentals["return_on_equity"] > 0.15 else (50 if fundamentals["return_on_equity"] > 0 else -100)

        fundamentals["fundamental_score"] = round(score / count) if count > 0 else 0

        return json.dumps(fundamentals, default=str)

    except Exception as e:
        return f"Error fetching fundamentals for {ticker}: {str(e)}"


@tool
def get_technicals(ticker: str) -> str:
    """
    Computes technical indicators for a stock ticker from 1 year of price history.
    Returns SMA50, SMA200, RSI, MACD, current price, and a technical score.
    Use this to understand price momentum and trend direction.
    """
    try:
        import yfinance as yf
        import pandas as pd

        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="1y")

        if hist.empty:
            return f"No price history found for {ticker}."

        closes = hist["Close"]

        # Simple Moving Averages
        sma50 = closes.rolling(window=50).mean()
        sma200 = closes.rolling(window=200).mean()

        # RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))

        # MACD
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd = float(ema12.iloc[-1] - ema26.iloc[-1])
        signal = float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])

        latest_price = float(closes.iloc[-1])
        latest_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None
        latest_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None

        # Score technicals
        score = 0
        count = 0

        if latest_sma200:
            count += 1
            score += 100 if latest_price > latest_sma200 else -100

        if rsi:
            count += 1
            score += -100 if rsi > 70 else (100 if rsi < 30 else 0)

        if macd and signal:
            count += 1
            score += 100 if macd > signal else -100

        if latest_sma50 and latest_sma200:
            count += 1
            score += 100 if latest_sma50 > latest_sma200 else -100

        technical_score = round(score / count) if count > 0 else 0

        technicals = {
            "ticker": ticker.upper(),
            "current_price": latest_price,
            "sma50": latest_sma50,
            "sma200": latest_sma200,
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "macd_signal": round(signal, 4),
            "price_vs_sma200": "above" if latest_sma200 and latest_price > latest_sma200 else "below",
            "golden_cross": latest_sma50 and latest_sma200 and latest_sma50 > latest_sma200,
            "rsi_signal": "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
            "macd_signal_direction": "bullish" if macd > signal else "bearish",
            "technical_score": technical_score,
        }

        return json.dumps(technicals, default=str)

    except Exception as e:
        return f"Error computing technicals for {ticker}: {str(e)}"


@tool
def run_sentiment(ticker: str) -> str:
    """
    Fetches recent news headlines for a stock ticker and runs FinBERT
    sentiment analysis on each headline. Returns sentiment scores,
    labels, and an aggregate sentiment score from -100 to +100.
    Use this to understand current market sentiment from recent news.
    """
    try:
        import yfinance as yf
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        stock = yf.Ticker(ticker.upper())

        # Fetch headlines
        headlines = []
        try:
            news_items = stock.news or []
            for item in news_items[:10]:
                title = ""
                if isinstance(item, dict) and "content" in item:
                    content = item["content"]
                    title = content.get("title", "") if isinstance(content, dict) else ""
                elif isinstance(item, dict):
                    title = item.get("title", "")
                if title:
                    headlines.append(title)
        except Exception:
            pass

        if not headlines:
            return json.dumps({
                "ticker": ticker.upper(),
                "headlines_analyzed": 0,
                "sentiment_score": 0,
                "message": "No recent news found"
            })

        # Run FinBERT
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        results = classifier(headlines, truncation=True, max_length=512)

        scored = []
        for headline, result in zip(headlines, results):
            label = result["label"].lower()
            conf = result["score"]
            numeric = int(conf * 100) if label == "positive" else (int(conf * -100) if label == "negative" else 0)
            scored.append({
                "headline": headline,
                "sentiment": label,
                "confidence": round(conf, 3),
                "score": numeric
            })

        aggregate_score = round(sum(r["score"] for r in scored) / len(scored))

        return json.dumps({
            "ticker": ticker.upper(),
            "headlines_analyzed": len(scored),
            "aggregate_sentiment_score": aggregate_score,
            "sentiment_breakdown": scored,
            "overall_sentiment": "positive" if aggregate_score > 20 else ("negative" if aggregate_score < -20 else "neutral")
        }, default=str)

    except Exception as e:
        return f"Error running sentiment for {ticker}: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ReAct Agent Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REACT_PROMPT = PromptTemplate.from_template("""
You are AlphaScope AI, an expert financial analysis agent.
Your job is to analyze a stock ticker by using your available tools,
then synthesize a structured investment report.

You have access to the following tools:
{tools}

Use this exact format:

Question: the stock ticker to analyze
Thought: think about what you need to do first
Action: the tool to use, must be one of [{tool_names}]
Action Input: the ticker symbol
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have all the data I need to write the report
Final Answer: A structured 4-paragraph investment report covering:
1. Fundamental Analysis - reference specific metrics and the fundamental score
2. Technical Analysis - reference price, SMAs, RSI, MACD and the technical score
3. Sentiment Analysis - reference headline count, aggregate score, overall sentiment
4. Investment Verdict - combine all three layers into a clear recommendation with key risks

Always call all three tools before writing the final answer.
Be specific with numbers. Frame as analysis only, not investment advice.

Question: {input}
Thought: {agent_scratchpad}
""")


def build_agent(groq_api_key: str) -> AgentExecutor:
    """
    Builds and returns a LangChain ReAct AgentExecutor.
    The agent autonomously decides to call get_fundamentals,
    get_technicals, and run_sentiment before synthesizing a report.
    """
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    tools = [get_fundamentals, get_technicals, run_sentiment]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )


def run_agentic_analysis(ticker: str, groq_api_key: str) -> str:
    """
    Entry point: runs the full agentic analysis for a ticker.
    Returns the final narrative report as a string.
    """
    agent_executor = build_agent(groq_api_key)
    result = agent_executor.invoke({"input": ticker.upper()})
    return result.get("output", "Agent did not return a result.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChromaDB RAG Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_vector_store(ticker: str, narrative: str, sentiment_results: list) -> Chroma:
    """
    Builds a ChromaDB vector store from:
    - The generated analyst narrative (chunked into paragraphs)
    - Individual news headlines with their sentiment labels

    Uses sentence-transformers (all-MiniLM-L6-v2) to embed documents.
    Returns a Chroma retriever ready for similarity search.
    """

    documents = []

    # Chunk the narrative into paragraphs
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        documents.append(Document(
            page_content=para,
            metadata={
                "source": "analyst_narrative",
                "ticker": ticker,
                "chunk_index": i
            }
        ))

    # Add each news headline as a document
    for item in sentiment_results:
        headline_text = f"{item['headline']} (Sentiment: {item['sentiment']}, Confidence: {item['confidence']})"
        documents.append(Document(
            page_content=headline_text,
            metadata={
                "source": "news_headline",
                "ticker": ticker,
                "sentiment": item["sentiment"],
                "score": item["score"]
            }
        ))

    # Embed using sentence-transformers (runs locally, no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Store in ChromaDB (in-memory, resets each session)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=f"alphascope_{ticker.lower()}"
    )

    return vector_store


def rag_chat(question: str, ticker: str, vector_store: Chroma, groq_api_key: str) -> str:
    """
    RAG-powered chat:
    1. Embeds the user question
    2. Retrieves top-3 most semantically similar chunks from ChromaDB
    3. Augments the prompt with retrieved context
    4. LLaMA 3.3 generates a grounded answer

    This is full RAG: Retrieve → Augment → Generate
    """

    # Step 1: Retrieve relevant chunks via cosine similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    relevant_docs = retriever.invoke(question)

    # Step 2: Build context string from retrieved chunks
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # Step 3: Augment prompt with retrieved context
    augmented_prompt = f"""You are AlphaScope AI, a financial analysis assistant for {ticker}.

Answer the user's question using ONLY the context below.
Be concise, reference specific numbers, and say so clearly if the context doesn't contain the answer.
Never give investment advice — frame everything as analysis only.

Retrieved Context:
{context}

User Question: {question}

Answer:"""

    # Step 4: Generate answer via LLaMA 3.3 through Groq
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": augmented_prompt}],
        max_tokens=500,
        temperature=0.3,
    )

    return response.choices[0].message.content
