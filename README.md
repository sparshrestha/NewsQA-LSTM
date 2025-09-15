
## How to Install and Run the Project

1. ``` git clone git@github.com:sparshrestha/NewsQA-LSTM.git ```
2. ``` cd NewsQA-LSTM ```
3. ```python -m venv venv```
4. ```for windows .\venv\Scripts\activate and for linux source venv/bin/activate```
5. ```pip install -r requirements.txt```
6. ```python main.py```

# NewsQA-LSTM

A work-in-progress project to build a **Question Answering system for news articles** using LSTM-based models.  
This system will:

1. **Ingest & clean** hundreds of news articles (with metadata).
2. **Retrieve** relevant passages using a BiLSTM-based retriever + FAISS/ScaNN index.
3. **Read & extract** short answers with an LSTM + attention reader.
4. **Return answers** with confidence and source citations (URL, headline, date).
5. **Fallback** to "not found" when confidence is low.


## Training Corpus
- **Source:** Crawl news articles from allowed RSS feeds / APIs / websites.
- **Size:** Hundreds of articles → at least **5,000 unique passages** after cleaning.
- **Preprocessing:**
  - Normalize Unicode & remove boilerplate text (ads, nav, etc.).
  - Split into **200–400 token passages** with ~50 token overlap.
  - Deduplicate near-identical passages using **shingling + MinHash** (target ≥10% reduction).
- **Metadata:** Each passage stores publisher, URL, headline, date, and detected entities.


## Tech Stack
- Python 3.10+
- PyTorch (for BiLSTM & reader)
- FAISS or ScaNN (for ANN search)
- BeautifulSoup / Newspaper3k (for news crawling & cleaning)


## Retrieval Model (BiLSTM)
- **Embeddings:** Word embeddings (GloVe / fastText) + optional char-CNN.
- **Encoder:** BiLSTM → pooling (mean/max/attentive) → fixed vector.
- **Training:** Contrastive / triplet loss with (query, positive, negative) triplets.
- **Index:** FAISS or ScaNN ANN index, storing passage IDs for citations.
- **Evaluation:**
  - **Recall@20:** Fraction of queries with relevant passage in top-20.
  - **MRR:** Mean reciprocal rank of correct passage.


## Reader Model (LSTM + Attention)
- **Architecture:** BiLSTM + attention over passage conditioned on question.
- **Output:** Start and end span predictions (softmax).
- **Training:**
  - Pretrain on SQuAD-style dataset (Wikipedia).
  - Fine-tune via distant supervision on news (when gold answers appear in passages).
- **Evaluation Metrics:**
  - **EM (Exact Match):** % answers exactly equal to gold.
  - **F1:** Overlap of predicted vs gold answer tokens.


## Inference Pipeline
1. Preprocess input question (tokenization, entities, date hints).
2. Retrieve **top-k=20** passages via retriever.
3. Re-rank with cross-encoder (question ⊕ passage), keep **top-m=5**.
4. Run reader on top-m passages to extract candidate answer spans.
5. Select highest-confidence span → return **answer + confidence + (URL, headline, date)**.
6. If confidence < τ (threshold), output **“not found”** and list top citations.


---
