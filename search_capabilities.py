"""This code is used for /searchgpt endpoint in fastapi. It searches DuckDuckGo, OpenWeatherMap, and returns fast, witty, fact-grounded responses."""
import os
import re
import json
import time
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

openai_org_id = os.getenv('OpenAI_ORG_ID')
openai_api_key = os.getenv('OPENAI_API_KEY')
Open_Weather_API_Key = os.getenv("OPENWEATHERMAP_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ============================================================
# LLM Initialization (Ultra-Fast with Token Limits)
# ============================================================
def get_search_llm(temperature=0.6, max_tokens=140):
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    # 1. Groq Cloud (Ultra-Fast)
    if (provider == "groq" or use_groq) and groq_api_key and not groq_api_key.startswith("your-"):
        try:
            from langchain_groq import ChatGroq
            print("[OK] SearchGPT using Groq API (Ultra-Fast Cloud)")
            return ChatGroq(
                temperature=temperature,
                model="llama-3.1-8b-instant",
                max_tokens=max_tokens,
                groq_api_key=groq_api_key
            )
        except Exception as e:
            print(f"[Warning] SearchGPT Groq error: {e}")

    # 2. Local Ollama (Optimized with num_predict)
    if provider == "ollama" or use_ollama or True:
        try:
            from langchain_community.chat_models import ChatOllama
            print(f"[OK] SearchGPT using Ollama ({ollama_model}) with max_tokens={max_tokens}")
            return ChatOllama(
                model=ollama_model,
                temperature=temperature,
                num_predict=max_tokens
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize ChatOllama in SearchGPT: {e}")

    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=ollama_model, temperature=temperature, num_predict=max_tokens)

llm = get_search_llm(temperature=0.6, max_tokens=140)

# ============================================================
# Fast Web Search & Tools
# ============================================================
def fast_web_search(query: str, max_results: int = 3) -> str:
    """Fast DuckDuckGo search with timeout."""
    try:
        ddgs = DDGS(timeout=5)
        results = []
        for r in ddgs.text(query, max_results=max_results):
            title = r.get('title', '')
            body = r.get('body', '')
            if title or body:
                results.append(f"- {title}: {body[:200]}")
        return "\n".join(results) if results else "No specific web results found."
    except Exception as e:
        print(f"[Web Search Warning]: {e}")
        return ""

def fast_weather_search(location: str) -> str:
    """Fetch weather via OpenWeatherMap or DuckDuckGo."""
    if Open_Weather_API_Key and not Open_Weather_API_Key.startswith("your-"):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={Open_Weather_API_Key}&units=metric"
            resp = requests.get(url, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                temp = data['main']['temp']
                desc = data['weather'][0]['description']
                city = data.get('name', location)
                return f"Current weather in {city}: {temp}°C, {desc}."
        except Exception as e:
            print(f"[Weather API Warning]: {e}")
    # Fallback to web search
    return fast_web_search(f"current weather in {location}")

# ============================================================
# SearchGPT Synthesis Prompt Chain
# ============================================================
search_prompt = PromptTemplate(
    template="""You are Search AI, a brilliant, witty, and helpful AI assistant created by Shaon Sikder.
You are entertaining, humorous like a late-night comedian, and always provide accurate, factual answers.

Facts / Search Results:
{search_context}

User Query: {query}
Search AI (Answer in 1 to 3 friendly, witty sentences with emojis):""",
    input_variables=["search_context", "query"]
)

search_chain = LLMChain(llm=llm, prompt=search_prompt)

# ============================================================
# Main Search Response Handler (Fast Dispatch)
# ============================================================
def get_response(message: str) -> str:
    """Processes user query, performs fast retrieval if needed, and generates concise response."""
    try:
        query = str(message).strip()
        print(f"SearchGPT Query: {query}")

        query_lower = query.lower()
        search_context = ""

        # 1. Weather Intent Check
        if "weather" in query_lower or "temperature" in query_lower or "forecast" in query_lower:
            loc_match = re.search(r'(?:in|for|at)\s+([a-zA-Z\s]+)', query, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else "New York"
            search_context = fast_weather_search(location)

        # 2. Real-Time Search / News / Fact Check Intent
        elif any(kw in query_lower for kw in ['who is', 'what is', 'latest', 'news', 'score', 'when did', 'where is', 'current', 'president', 'price of bitcoin', 'stock']):
            search_context = fast_web_search(query, max_results=2)

        # 3. Fast Synthesis Call
        response = search_chain.run(
            search_context=search_context or "General knowledge query.",
            query=query
        )

        cleaned_res = response.replace("Search AI:", "").strip()
        print(f"SearchGPT Response: {cleaned_res}")
        return cleaned_res

    except Exception as e:
        error_msg = str(e)
        print(f"[SearchGPT Error]: {error_msg}")
        return f"Hey there! 🤖 I encountered a quick hiccup: {error_msg}. Please try asking again!"