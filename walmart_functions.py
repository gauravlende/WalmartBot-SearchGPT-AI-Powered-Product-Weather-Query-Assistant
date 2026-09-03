"""This code is used for /walmartbot endpoint in fastapi. It processes messages, searches the Walmart catalog/vector database, and returns sales responses with product links and prices."""
import os
import re
import json
import time
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Union, Callable
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
OPENAI_ORG_ID = os.getenv('OpenAI_ORG_ID') or 'OpenAI_ORG_ID'
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') or 'PINECONE_API_KEY'

from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain.chains.base import Chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

# ============================================================
# LLM Initialization (Ultra-Fast with Token Limits)
# ============================================================
def get_llm(temperature=0.2, max_tokens=150):
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    # 1. Groq Cloud (Ultra-Fast ~300ms)
    if (provider == "groq" or use_groq) and groq_api_key and not groq_api_key.startswith("your-"):
        try:
            from langchain_groq import ChatGroq
            print("[OK] WalmartBot using Groq API (Ultra-Fast Cloud)")
            return ChatGroq(
                temperature=temperature,
                model="llama-3.1-8b-instant",
                max_tokens=max_tokens,
                groq_api_key=groq_api_key
            )
        except Exception as e:
            print(f"[Warning] Groq initialization failed: {e}")

    # 2. Local Ollama (Optimized with num_predict for quick responses)
    if provider == "ollama" or use_ollama or True:
        try:
            from langchain_community.chat_models import ChatOllama
            print(f"[OK] WalmartBot using Ollama ({ollama_model}) with max_tokens={max_tokens}")
            return ChatOllama(
                model=ollama_model,
                temperature=temperature,
                num_predict=max_tokens
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize ChatOllama: {e}")

    # 3. OpenAI Fallback
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not openai_key.startswith("your-") and not openai_key.startswith("sk-proj-"):
        try:
            from langchain.chat_models import ChatOpenAI
            print("[Info] WalmartBot using ChatOpenAI (gpt-3.5-turbo)")
            return ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            print(f"[Warning] OpenAI init error: {e}")

    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=ollama_model, temperature=temperature, num_predict=max_tokens)

llm = get_llm(temperature=0.2, max_tokens=150)

# ============================================================
# High-Speed In-Memory Walmart Product Catalog Engine (~3ms)
# ============================================================
class FastWalmartCatalog:
    """Inverted index catalog search engine over WMT_Grocery_Data.csv."""
    def __init__(self, csv_path="Data/WMT_Grocery_Data.csv"):
        self.df = None
        self.index_map = defaultdict(set)
        if os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                for idx, name in enumerate(self.df['PRODUCT_NAME']):
                    if pd.isna(name):
                        continue
                    words = re.findall(r'[a-zA-Z0-9]+', str(name).lower())
                    for w in words:
                        if len(w) > 2:
                            self.index_map[w].add(idx)
                for idx, cat in enumerate(self.df['CATEGORY']):
                    if pd.isna(cat):
                        continue
                    words = re.findall(r'[a-zA-Z0-9]+', str(cat).lower())
                    for w in words:
                        if len(w) > 2:
                            self.index_map[w].add(idx)
                print(f"[FastWalmartCatalog] Loaded {len(self.df)} products, indexed {len(self.index_map)} keywords.")
            except Exception as e:
                print(f"[FastWalmartCatalog] Error indexing {csv_path}: {e}")
        else:
            print(f"[FastWalmartCatalog] Warning: {csv_path} not found.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.df is None or not query:
            return []
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', query) if len(w) > 2]
        stop_words = {
            "what", "where", "which", "want", "need", "like", "give", "show",
            "have", "with", "from", "that", "this", "help", "please", "price",
            "cost", "much", "buy", "sell", "looking", "some", "item", "product",
            "walmart", "tell", "about", "your", "good", "best", "store"
        }
        search_words = [w for w in words if w not in stop_words] or words

        score_map = defaultdict(int)
        for w in search_words:
            if w in self.index_map:
                for row_idx in self.index_map[w]:
                    score_map[row_idx] += 1

        if not score_map:
            return []

        top_indices = sorted(score_map.keys(), key=lambda idx: score_map[idx], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "name": str(row['PRODUCT_NAME']),
                "price": float(row['PRICE_CURRENT']) if not pd.isna(row['PRICE_CURRENT']) else 0.0,
                "category": str(row.get('CATEGORY', '')),
                "url": str(row['PRODUCT_URL'])
            })
        return results

catalog = FastWalmartCatalog()

# ============================================================
# Knowledge Base Tool Wrapper
# ============================================================
class KnowledgeBaseTool:
    name = "ProductInfoSearch"
    description = "Searches the Walmart grocery catalog for products, prices, and URLs."
    sources_list: List[str] = []

    def run(self, query: str) -> str:
        products = catalog.search(query, top_k=3)
        if not products:
            return "No specific matching products found in the catalog."
        
        self.sources_list = [f"{p['name']} (${p['price']}) - {p['url']}" for p in products]
        lines = []
        for p in products:
            lines.append(f"- {p['name']}: ${p['price']} (Link: {p['url']})")
        return "\n".join(lines)

knowledge_base = KnowledgeBaseTool()

def get_tools(product_catalog=None):
    return [knowledge_base]

# ============================================================
# Conversation Stages & Chains
# ============================================================
conversation_stages = {
    '1': "Introduction: Greeting customer, introducing Walmart SalesBot, and asking what product they are looking for.",
    '2': "Show/Search Products Information: Assisting customer with grocery products, features, exact prices, and purchase recommendations.",
    '3': "Others Queries (Not Related): Friendly, witty Walmart representative answering general questions while steering back to Walmart products."
}

class StageAnalyzerChain:
    """Fast rule-and-intent stage analyzer (0ms latency)."""
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False):
        return cls()

    def run(self, conversation_history: str = "", current_conversation_stage: str = "1", **kwargs) -> str:
        if not conversation_history or len(conversation_history.strip()) == 0:
            return "1"
        history_lower = conversation_history.lower()
        product_indicators = [
            'buy', 'want', 'need', 'price', 'cost', 'how much', 'looking for',
            'product', 'item', 'have', 'sell', 'grocery', 'food', 'snack', 'drink',
            'organic', 'milk', 'cheese', 'bread', 'apple', 'tomato', 'potato',
            'water', 'chips', 'cookie', 'meat', 'chicken', 'coffee', 'tea', 'cereal'
        ]
        if any(ind in history_lower for ind in product_indicators):
            return "2"
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'who are you', 'what is your name']
        if any(g in history_lower for g in greetings) and len(conversation_history.split('\n')) <= 2:
            return "1"
        return "2" if "2" in current_conversation_stage else "3"

class SalesConversationChain:
    """Direct single-pass sales conversation utterance chain."""
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False):
        sales_prompt_template = """You are {salesperson_name}, a friendly, helpful, and energetic Walmart sales representative.
Company: {company_name}
Company Goal: {company_values}

Conversation Stage: {conversation_stage}

Available Product In-Stock Matches (Use these exact names and prices when answering):
{product_context}

Previous Conversation History:
{conversation_history}

{salesperson_name} (Keep response short, natural, friendly, 1 to 3 sentences, mention prices clearly):"""

        prompt = PromptTemplate(
            template=sales_prompt_template,
            input_variables=[
                "salesperson_name",
                "company_name",
                "company_values",
                "conversation_stage",
                "product_context",
                "conversation_history"
            ]
        )
        return LLMChain(prompt=prompt, llm=llm, verbose=verbose)

# ============================================================
# SalesGPT Controller (Ultra-Fast Single-Pass Pipeline)
# ============================================================
class SalesGPT(BaseModel):
    conversation_history: List[str] = []
    current_conversation_stage: str = '1'
    sales_conversation_utterance_chain: Any = None
    stage_analyzer_chain: Any = None
    salesperson_name: str = "Walmart Bot"
    salesperson_role: str = "Sales Representative"
    company_name: str = "Walmart"
    company_business: str = "Walmart Inc. is an American retail corporation providing groceries, general merchandise, and value."
    company_values: str = "Save money so you can live better."
    conversation_purpose: str = "Help customers find products and prices."
    conversation_type: str = "chat"
    use_tools: bool = True
    product_catalog: str = "Data/WMT_Grocery_Data.csv"
    conversation_stage_dict: Dict = conversation_stages

    class Config:
        arbitrary_types_allowed = True

    def retrieve_conversation_stage(self, key: str) -> str:
        return self.conversation_stage_dict.get(key, self.conversation_stage_dict['1'])

    def seed_agent(self):
        self.current_conversation_stage = self.retrieve_conversation_stage('1')
        self.conversation_history = []
        knowledge_base.sources_list = []

    def determine_conversation_stage(self):
        stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage
        )
        self.current_conversation_stage = self.retrieve_conversation_stage(stage_id)
        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input: str):
        cleaned_input = f"User: {human_input.strip()} <END_OF_TURN>"
        if len(self.conversation_history) >= 8:
            self.conversation_history.pop(0)
        self.conversation_history.append(cleaned_input)

    def step(self) -> str:
        return self._call()

    def _call(self) -> str:
        # 1. Extract latest user query for product lookup
        user_query = ""
        for msg in reversed(self.conversation_history):
            if msg.startswith("User:"):
                user_query = msg.replace("User:", "").replace("<END_OF_TURN>", "").strip()
                break

        # 2. Fast catalog lookup (~3ms)
        products = catalog.search(user_query, top_k=3)
        if products:
            knowledge_base.sources_list = [f"{p['name']} (${p['price']}) - {p['url']}" for p in products]
            product_context = "\n".join([f"- {p['name']}: ${p['price']} (Link: {p['url']})" for p in products])
        else:
            knowledge_base.sources_list = []
            product_context = "No specific catalog matches found for this query."

        # 3. Single-pass LLM generation (~1-3s on Groq, ~10s on CPU Ollama)
        try:
            ai_message = self.sales_conversation_utterance_chain.run(
                salesperson_name=self.salesperson_name,
                company_name=self.company_name,
                company_values=self.company_values,
                conversation_stage=self.current_conversation_stage,
                product_context=product_context,
                conversation_history="\n".join(self.conversation_history[-4:])
            )
        except Exception as e:
            print(f"[Error in SalesGPT LLM call]: {e}")
            if products:
                p = products[0]
                ai_message = f"Hello! We have {p['name']} available for ${p['price']}. Let me know if you would like more details!"
            else:
                ai_message = "Hello! Welcome to Walmart. How can I help you find grocery products or deals today?"

        cleaned_response = ai_message.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "").strip()
        print(f"{self.salesperson_name}: {cleaned_response}")

        self.conversation_history.append(f"{self.salesperson_name}: {cleaned_response} <END_OF_TURN>")
        return cleaned_response

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        stage_analyzer = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose)

        return cls(
            stage_analyzer_chain=stage_analyzer,
            sales_conversation_utterance_chain=utterance_chain,
            **kwargs
        )

# Default configuration
config = dict(
    salesperson_name="Walmart Bot",
    salesperson_role="Sales Representative",
    company_name="Walmart",
    company_business="Walmart Inc. is an American retail corporation that operates department stores and grocery stores.",
    company_values="Save money, live better.",
    conversation_purpose="Find out what products the customer is looking to buy and provide prices and links.",
    conversation_history=[],
    conversation_type="chat",
    conversation_stage=conversation_stages['1'],
    use_tools=True,
    product_catalog="Data/WMT_Grocery_Data.csv"
)
