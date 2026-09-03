import time
import pandas as pd
import re
from collections import defaultdict

t0 = time.time()
df = pd.read_csv('Data/WMT_Grocery_Data.csv')
print(f"Loaded CSV in {round(time.time() - t0, 3)}s, total rows: {len(df)}")

# Build inverted word index
t_idx = time.time()
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

print(f"Built inverted index in {round(time.time() - t_idx, 4)}s (unique words: {len(index_map)})")

def fast_search_products(query, top_k=3):
    words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', query) if len(w) > 2]
    if not words:
        return []
    
    score_map = defaultdict(int)
    for w in words:
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
            'name': row['PRODUCT_NAME'],
            'price': row['PRICE_CURRENT'],
            'category': row['CATEGORY'],
            'url': row['PRODUCT_URL'],
            'score': score_map[idx]
        })
    return results

t1 = time.time()
res = fast_search_products('I want to buy organic tomato or cheese')
t_elapsed = time.time() - t1
print(f"Search completed in {round(t_elapsed * 1000, 3)} ms ({round(t_elapsed, 6)}s)")
print(f"Results found: {len(res)}")
for r in res:
    print(f" - {r['name']} ${r['price']}: {r['url']}")
