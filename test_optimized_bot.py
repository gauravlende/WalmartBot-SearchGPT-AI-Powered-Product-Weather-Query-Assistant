import time
import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Load catalog
df = pd.read_csv('Data/WMT_Grocery_Data.csv')
index_map = defaultdict(set)
for idx, name in enumerate(df['PRODUCT_NAME']):
    if pd.isna(name):
        continue
    words = re.findall(r'[a-zA-Z0-9]+', str(name).lower())
    for w in words:
        if len(w) > 2:
            index_map[w].add(idx)

for idx, cat in enumerate(df['CATEGORY']):
    if pd.isna(cat):
        continue
    words = re.findall(r'[a-zA-Z0-9]+', str(cat).lower())
    for w in words:
        if len(w) > 2:
            index_map[w].add(idx)

def fast_search(query, top_k=3):
    words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', query) if len(w) > 2]
    stop_words = {"what", "where", "which", "want", "need", "like", "give", "show", "have", "with", "from", "that", "this", "help", "please", "price", "cost", "much", "buy", "sell", "looking", "some"}
    search_words = [w for w in words if w not in stop_words] or words
    score_map = defaultdict(int)
    for w in search_words:
        if w in index_map:
            for row_idx in index_map[w]:
                score_map[row_idx] += 1
    if not score_map:
        return []
    top_indices = sorted(score_map.keys(), key=lambda idx: score_map[idx], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            'name': str(row['PRODUCT_NAME']),
            'price': float(row['PRICE_CURRENT']) if not pd.isna(row['PRICE_CURRENT']) else 0.0,
            'url': str(row['PRODUCT_URL'])
        })
    return results

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(model="llama3.2:latest", temperature=0.2, num_predict=120)

prompt_template = PromptTemplate(
    template="""You are Walmart SalesBot, an enthusiastic and helpful Walmart sales representative.
Your goal is to help the customer find grocery products, mention accurate prices and details, and assist them friendly and concisely.

Product Catalog Results:
{product_context}

Conversation History:
{history}

Customer: {user_input}
Walmart SalesBot (Keep answer short and friendly, under 3 sentences):""",
    input_variables=["product_context", "history", "user_input"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# Test query 1
query = "Hi, I am looking to buy some organic cherry tomatoes and ketchup. Do you have them?"
t0 = time.time()
products = fast_search(query, top_k=2)
t_search = time.time() - t0

if products:
    p_context = "\n".join([f"- {p['name']}: ${p['price']} (Link: {p['url']})" for p in products])
else:
    p_context = "No specific matching products found in catalog."

t1 = time.time()
response = chain.run(product_context=p_context, history="", user_input=query)
t_llm = time.time() - t1
total_time = time.time() - t0

print("=== TEST RESULTS ===")
print(f"Search time: {round(t_search*1000, 2)} ms")
print(f"LLM Generation time: {round(t_llm, 2)} s")
print(f"Total Response time: {round(total_time, 2)} s")
print("\nResponse:")
print(response)
print("\nSources:")
for p in products:
    print(f"- {p['name']} (${p['price']}): {p['url']}")
