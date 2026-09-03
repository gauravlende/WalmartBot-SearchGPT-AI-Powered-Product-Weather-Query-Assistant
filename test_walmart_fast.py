import time
import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# 1. Fast Product Search Engine
class FastWalmartCatalog:
    def __init__(self, csv_path="Data/WMT_Grocery_Data.csv"):
        self.df = None
        self.index_map = defaultdict(set)
        if os.path.exists(csv_path):
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
        else:
            print(f"[FastWalmartCatalog] Warning: {csv_path} not found.")

    def search(self, query: str, top_k: int = 3):
        if self.df is None or not query:
            return []
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', query) if len(w) > 2]
        # Ignore generic stop words for product search
        stop_words = {"what", "where", "which", "want", "need", "like", "give", "show", "have", "with", "from", "that", "this", "help", "please", "price", "cost", "much", "buy"}
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

# Test search
t0 = time.time()
res = catalog.search("Can I get some organic milk and cereal?")
print(f"Search time: {round((time.time() - t0)*1000, 2)}ms")
for r in res:
    print(" -", r['name'], f"${r['price']}", r['url'])
