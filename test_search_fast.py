import time
import os
import re
from duckduckgo_search import DDGS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.6,
    num_predict=100
)

def fast_web_search(query: str, max_results=3):
    try:
        ddgs = DDGS()
        results = []
        for r in ddgs.text(query, max_results=max_results):
            title = r.get('title', '')
            body = r.get('body', '')
            results.append(f"- {title}: {body[:250]}")
        return "\n".join(results)
    except Exception as e:
        return f"Web search error: {e}"

search_prompt = PromptTemplate(
    template="""You are Search AI, a witty, helpful AI assistant developed by Shaon Sikder.
Search Results / Facts:
{search_context}

User Query: {query}
Search AI (Be witty, friendly with emojis, under 3 sentences):""",
    input_variables=["search_context", "query"]
)

chain = LLMChain(llm=llm, prompt=search_prompt)

t0 = time.time()
query = "What is the latest score in soccer or premier league?"
web_info = fast_web_search(query, max_results=2)
t_web = time.time() - t0

t1 = time.time()
answer = chain.run(search_context=web_info, query=query)
t_llm = time.time() - t1
total = time.time() - t0

print("=== SearchGPT Test ===")
print(f"Web search time: {round(t_web, 2)}s")
print(f"LLM time: {round(t_llm, 2)}s")
print(f"Total time: {round(total, 2)}s")
print("Response:\n", answer)
