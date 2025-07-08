# src/rag_pipeline.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List
from langchain.prompts import PromptTemplate

# Load FAISS index and metadata
def load_faiss_index():
    index = faiss.read_index("vector_store/index.faiss")
    with open("vector_store/index.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load embedding and generation models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2")  # Replace with better LLM if available

# Prompt template
prompt_template = PromptTemplate.from_template(
    """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""
)

# RAG query function
def rag_query(question: str, k: int = 5) -> dict:
    index, metadata = load_faiss_index()

    # Step 1: Embed the question
    q_embedding = embedding_model.encode([question]).astype("float32")

    # Step 2: Retrieve top-k similar vectors
    D, I = index.search(q_embedding, k)
    retrieved = [metadata[i] for i in I[0]]

    # Step 3: Concatenate context
    context = "\n---\n".join([r["text"] for r in retrieved])

    # Step 4: Format the prompt
    prompt = prompt_template.format(context=context, question=question)

    # Step 5: Generate answer
    response = generator(prompt, max_length=512, do_sample=True)[0]["generated_text"]

    return {
        "question": question,
        "generated_answer": response,
        "retrieved_contexts": retrieved
    }
