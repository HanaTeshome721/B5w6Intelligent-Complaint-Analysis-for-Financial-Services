# src/embedding_pipeline.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load cleaned complaint data
df = pd.read_csv('data/filtered_complaints.csv')

# Step 2: Filter to relevant products and valid narratives
target_products = [
    'Credit card', 'Personal loan', 'Buy Now, Pay Later',
    'Savings account', 'Money transfer, virtual currency, or money service'
]

df_filtered = df[
    df['Product'].isin(target_products) &
    df['Consumer complaint narrative'].notna()
]

# Generate unique ID if not present
if 'complaint_id' not in df_filtered.columns:
    df_filtered['complaint_id'] = df_filtered.index.astype(str)

# Step 3: Initialize text chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Step 4: Chunk text and prepare metadata
docs = []
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    chunks = text_splitter.split_text(str(row['Consumer complaint narrative']))
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "metadata": {
                "complaint_id": row['complaint_id'],
                "product": row['Product'],
                "chunk_index": i
            }
        })

# Step 5: Embed the text using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [doc['text'] for doc in docs]
embeddings = model.encode(texts, show_progress_bar=True)

# Step 6: Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Step 7: Save FAISS index and metadata
os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/index.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Embedding pipeline complete. Index saved in vector_store/")
