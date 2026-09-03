import os
import sys
from dotenv import load_dotenv
load_dotenv()

from search_capabilities import *
from walmart_functions import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Walmart SalesBot & SearchGPT API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Sales Agent
sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)

class ConversationRequest(BaseModel):
    messages: List[str]

class ConversationResponse(BaseModel):
    messages: List[str]
    sources: Optional[List[str]] = []
    error: Optional[bool] = False

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    messages: List[str]
    error: Optional[bool] = False

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Walmart SalesBot & SearchGPT API is running fast!"}

@app.post("/walmartbot")
def handle_conversation(request: ConversationRequest):
    try:
        if not request.messages:
            sales_agent.seed_agent()
            sales_agent.determine_conversation_stage()
            ai_message = "Hello! Welcome to Walmart. I'm your sales assistant. How can I help you find grocery products or prices today?"
            return {
                "messages": [ai_message],
                "sources": [],
                "error": False
            }

        last_message = request.messages[-1]
        print(f"\n[WalmartBot] User: {last_message}")
        
        sales_agent.human_step(last_message)
        sales_agent.determine_conversation_stage()
        ai_message = sales_agent.step()

        sources = list(knowledge_base.sources_list)
        knowledge_base.sources_list = []

        print(f"[WalmartBot] Response: {ai_message}")
        if sources:
            print(f"[WalmartBot] Sources: {sources}")

        return {
            "messages": [str(ai_message)],
            "sources": sources,
            "error": False
        }

    except Exception as e:
        error_str = str(e)
        print(f"[WalmartBot Error]: {error_str}")
        return {
            "messages": [f"⚠️ Error: {error_str}"],
            "sources": [],
            "error": True
        }

@app.post("/searchgpt")
def handle_chat(request: ChatRequest):
    try:
        if not request.text or not request.text.strip():
            return {
                "messages": ["Please enter a question in the message box!"],
                "error": False
            }

        input_text = request.text.strip()
        print(f"\n[SearchGPT] User: {input_text}")
        response = get_response(input_text)
        print(f"[SearchGPT] Response: {response}")

        return {
            "messages": [str(response)],
            "error": False
        }

    except Exception as e:
        error_str = str(e)
        print(f"[SearchGPT Error]: {error_str}")
        return {
            "messages": [f"⚠️ Error: {error_str}"],
            "error": True
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)