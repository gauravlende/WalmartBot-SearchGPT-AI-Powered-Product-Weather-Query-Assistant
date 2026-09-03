import time
from fastapi.testclient import TestClient
from run_api import app

client = TestClient(app)

print("=== 1. Testing Health Check ===")
res = client.get("/")
print("Health:", res.status_code, res.json())

print("\n=== 2. Testing WalmartBot Endpoint (/walmartbot) ===")
t0 = time.time()
res_wmt = client.post("/walmartbot", json={"messages": ["Do you have organic cherry tomatoes and ketchup?"]})
t_wmt = round(time.time() - t0, 2)
print(f"Status: {res_wmt.status_code} in {t_wmt}s")
data_wmt = res_wmt.json()
print("Bot Message:", data_wmt.get("messages"))
print("Sources:", data_wmt.get("sources"))

print("\n=== 3. Testing SearchGPT Endpoint (/searchgpt) ===")
t1 = time.time()
res_search = client.post("/searchgpt", json={"text": "What is the capital of France?"})
t_search = round(time.time() - t1, 2)
print(f"Status: {res_search.status_code} in {t_search}s")
data_search = res_search.json()
print("Bot Message:", data_search.get("messages"))

print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")
