# 🧠 Task 2: Text Chunking, Embedding & Vector Store Indexing

This task converts cleaned complaint narratives into semantically meaningful vector representations to enable efficient retrieval in the Retrieval-Augmented Generation (RAG) chatbot pipeline.

---

## ✅ Objective

To prepare unstructured customer complaint narratives for semantic search by:

- Splitting long texts into manageable chunks
- Generating embeddings for each chunk using a transformer model
- Indexing embeddings in a FAISS vector store
- Saving relevant metadata for traceability and filtering

---

## 📁 Input Data

- `data/filtered_complaints.csv`  
  Cleaned complaint data from Task 1 with at least the following columns:
  - `Product`
  - `Consumer complaint narrative`
  - `complaint_id` (auto-generated if not present)

---

## ⚙️ Components

### 1. **Chunking Strategy**
- Implemented using LangChain's `RecursiveCharacterTextSplitter`
- **Chunk Size**: 500 characters  
- **Chunk Overlap**: 100 characters  
- Purpose: Preserve semantic context for better embedding and search relevance

### 2. **Embedding Model**
- Model: `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/)
- Dimensionality: 384
- Justification:
  - Compact and efficient
  - Strong performance on semantic similarity tasks
  - Well-suited for large-scale indexing and real-time inference

### 3. **Vector Indexing**
- Backend: FAISS (`IndexFlatL2`)
- Each embedding is stored along with:
  - `complaint_id`
  - `product`
  - `chunk_index`

---

## 🛠️ How to Run

```bash
python src/embedding_pipeline.py
