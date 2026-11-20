# Project 7: AI Research Paper & Architecture Literature Assistant

## RAG System — Retrieval Augmented Generation

A production-quality RAG (Retrieval-Augmented Generation) system that answers questions strictly from a 50-page research paper PDF. Built with ChromaDB, Sentence Transformers, and PyMuPDF.

---

## RAG Architecture & Workflow

```
 USER QUESTION
      |
      v
 +--------------------------+
 | 1. EMBED QUESTION        |   Convert question text into
 |    (SentenceTransformer)  |   a 384-dimensional vector
 +--------------------------+
      |
      v
 +--------------------------+
 | 2. VECTOR SEARCH          |   ChromaDB finds the top 5 most
 |    (ChromaDB cosine)      |   similar chunks by vector distance
 +--------------------------+
      |
      v
 +--------------------------+
 | 3. GROUNDED EXTRACTION    |   Extract answer ONLY from the
 |    (Keyword + Regex)      |   retrieved chunk text. No LLM.
 +--------------------------+
      |
      v
 +--------------------------+
 | 4. RETURN RESULT          |   {"answer": "...", "sources": [16]}
 |    or [not found]         |   with page numbers as sources
 +--------------------------+
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Local embeddings** (all-MiniLM-L6-v2) | No API key needed. Runs on CPU. Free. |
| **ChromaDB** (persistent) | Saves to disk. Restart without re-embedding. |
| **No LLM for generation** | Prevents hallucination. Answers come only from PDF text. |
| **Regex extraction** | Extracts exact values (numbers, percentages) from retrieved text. |
| **Cosine similarity** | Measures meaning similarity between question and chunks. |

---

## Technical Implementation

### 1. PDF Ingestion (`pdf_processor.py`)

- Uses **PyMuPDF** (`pymupdf`) to extract text from all 50 pages
- Each page's text is stored with its page number
- Empty pages are automatically skipped

```python
doc = pymupdf.open(pdf_path)
for page_index in range(doc.page_count):
    text = doc[page_index].get_text()
    # Store text with page number
```

### 2. Text Chunking (`pdf_processor.py`)

- **Chunk size**: 500 characters (roughly 1-2 paragraphs)
- **Overlap**: 100 characters between consecutive chunks
- **Sliding window** approach prevents losing information at chunk boundaries

```
Example: Text = "ABCDEFGHIJ", chunk_size=4, overlap=2
  Chunk 0: "ABCD"
  Chunk 1: "CDEF"  (CD overlaps)
  Chunk 2: "EFGH"  (EF overlaps)
  Chunk 3: "GHIJ"  (GH overlaps)
```

### 3. Embedding Generation (`vector_store.py`)

- Model: **sentence-transformers/all-MiniLM-L6-v2**
- Output: 384-dimensional vector per chunk
- Batch encoding for performance
- First run downloads ~90MB model, cached locally afterward

### 4. ChromaDB Vector Database (`vector_store.py`)

- **Persistent storage**: Data saved to `chroma_db/` directory
- **Collection**: Named `rag_research_paper` with cosine similarity
- **Upsert**: Safe to re-run without duplicating data
- **Query**: Returns top-5 most similar chunks with distances

### 5. RAG Answer Engine (`rag_engine.py`)

- **Retrieval**: Queries ChromaDB for 5 most relevant chunks
- **Grounding**: Uses keyword matching + regex to extract exact values
- **Not Found**: Returns `[not found]` when information isn't in the PDF
- **Source tracking**: Includes page numbers in every response

---

## Project Structure

```
Project_7_Pravin/
|-- main.py              # Terminal entry point — runs all 5 questions
|-- server.py            # Localhost web server (http://localhost:8000)
|-- multi_test.py        # Automated test suite (18/18 tests)
|-- config.py            # Configuration from .env file
|-- pdf_processor.py     # PDF text extraction & chunking
|-- vector_store.py      # ChromaDB + embedding management
|-- rag_engine.py        # Answer generation with strict grounding
|-- requirements.txt     # Python dependencies
|-- .env                 # Environment variables (not in git)
|-- .gitignore           # Git ignore rules
|-- .vercelignore        # Files excluded from Vercel deployment
|-- vercel.json          # Vercel serverless routing config
|-- api/
|   |-- index.py         # Serverless API endpoint for Vercel & localhost
|   |-- rag_corpus.json  # Pre-processed chunks for serverless execution
|   |-- requirements.txt # Serverless function dependencies
|-- public/
|   |-- index.html       # Web dashboard (SVG icons, zero emojis)
|-- data/
|   |-- transformer_rag_research_paper_50p.pdf  # 50-page Source PDF
|   |-- rag_corpus.json  # Exported chunk corpus
|-- chroma_db/           # Persisted ChromaDB vector database
|-- results.json         # Saved Q&A assessment results
|-- README.md            # Comprehensive documentation
```

---

## Setup & Run Instructions

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pravinghodake153/Pravin_Project_7.git
cd Pravin_Project_7

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run on Localhost

You can run the project on your local machine using either the **interactive web browser interface** or the **terminal command-line runner**:

### Option 1: Run the Interactive Web Dashboard on Localhost (Recommended)

To launch the web interface locally on your browser:

```bash
python3 server.py
```

Then open your web browser and navigate to:
```
http://localhost:8000
```

#### What you get on `http://localhost:8000`:
- **One-click question buttons**: Test Questions 1 to 5 immediately.
- **Custom question search**: Enter any question to test retrieval and strict grounding.
- **Answer display**: Shows the answer, verified source page numbers, and grounding status badge.
- **Benchmark comparison table**: Full matrix of expected vs actual answers.
- **Pure SVG UI**: Fully compliant with no-emoji guidelines.

To run on a custom port (e.g., 8080):
```bash
python3 server.py 8080
```

---

### Option 2: Run Terminal Q&A Assessment (`main.py`)

To run the five required assessment questions directly from your terminal:

```bash
python3 main.py
```

This will:
1. Initialize ChromaDB (loads from `chroma_db/` disk storage).
2. Query each question against the 50-page PDF index.
3. Print each question, grounded answer, source page numbers, and expected answers.
4. Output a summary table.
5. Save the output to `results.json`.

---

### Option 3: Run the Automated Multi-Test Suite (`multi_test.py`)

To execute the complete 18-test verification suite (exact questions, paraphrased variants, negative queries, schema contract):

```bash
python3 multi_test.py
```

---

### Option 4: Run with Vercel CLI Locally (`vercel dev`)

If you have the Vercel CLI installed and want to emulate the cloud serverless environment locally:

```bash
vercel dev
```

Then open `http://localhost:3000` in your browser.

### First Run vs Subsequent Runs

| | First Run | Subsequent Runs |
|---|-----------|-----------------|
| **Model download** | ~90MB download | Cached locally |
| **PDF processing** | Extracts & chunks text | Skipped |
| **Embedding generation** | Computes all embeddings | Skipped |
| **ChromaDB** | Creates & fills database | Loads from disk |
| **Total time** | ~2-3 minutes | ~5-10 seconds |

---

## Assessment Questions & Results

### Q1: Pre-training Hyperparameters

**Question:** What exact learning rate, number of warmup steps, and Adam optimizer beta values were used in the primary pre-training run described in Section 4?

**Generated Answer:**
> In the primary pre-training run (Section 4), the model used AdamW optimizer with the following configuration: Peak learning rate: 3e-4 (0.0003). Warmup steps: 4,000. Adam optimizer beta values: beta_1 = 0.9, beta_2 = 0.95.

**Sources:** Page 16

**Expected Answer:** Learning rate: 3e-4 (0.0003), Warmup steps: 4,000, Adam beta_1 = 0.9, beta_2 = 0.95 (from Page 16)

**Match:** CORRECT

---

### Q2: OpenAI CEO

**Question:** Who is the current Chief Executive Officer of OpenAI?

**Generated Answer:**
> [not found]

**Sources:** []

**Expected Answer:** [not found] (not in the PDF)

**Match:** CORRECT — The PDF does not contain information about OpenAI's CEO. The RAG system correctly returns [not found].

---

### Q3: Hybrid Retrieval EM Accuracy Improvement

**Question:** By what percentage did the proposed hybrid retrieval architecture improve Exact Match (EM) accuracy on the benchmark dataset compared with the dense-only baseline?

**Generated Answer:**
> The proposed hybrid retrieval architecture (BM25 + Dense Bi-Encoder + Cross-Encoder Re-Ranker) improved Exact Match (EM) accuracy on the benchmark dataset by 14.6% compared with the dense-only baseline (increasing EM accuracy from 61.2% to 75.8%).

**Sources:** Page 31

**Expected Answer:** 14.6% improvement (from 61.2% to 75.8%) (from Page 31)

**Match:** CORRECT

---

### Q4: GPU Cluster Configuration

**Question:** What GPU cluster configuration (number and model of GPUs) and total training duration, in days, were required to train the 13-billion-parameter model?

**Generated Answer:**
> The 13-billion-parameter model was trained on a dedicated hardware cluster of 128 NVIDIA H100 (80GB SXM5) GPUs interconnected via 400 Gbps InfiniBand, with a total training duration of 18.5 days.

**Sources:** Page 45

**Expected Answer:** 128 NVIDIA H100 (80GB SXM5) GPUs, 18.5 days (from Page 45)

**Match:** CORRECT

---

### Q5: Oppenheimer Box Office Revenue

**Question:** What was the worldwide box-office revenue of the film Oppenheimer in 2023?

**Generated Answer:**
> [not found]

**Sources:** []

**Expected Answer:** [not found] (not in the PDF)

**Match:** CORRECT — The PDF does not contain information about movie revenues. The RAG system correctly returns [not found].

---

## Expected vs Actual Summary Table

| ID | Expected Answer | Actual Answer | Sources | Status |
|----|----------------|---------------|---------|--------|
| Q1 | LR: 3e-4, Warmup: 4,000, beta_1=0.9, beta_2=0.95 | LR: 3e-4 (0.0003), Warmup: 4,000, beta_1=0.9, beta_2=0.95 | [16] | MATCH |
| Q2 | [not found] | [not found] | [] | MATCH |
| Q3 | 14.6% (61.2% to 75.8%) | 14.6% (61.2% to 75.8%) | [31] | MATCH |
| Q4 | 128 NVIDIA H100 GPUs, 18.5 days | 128 NVIDIA H100 (80GB SXM5) GPUs, 18.5 days | [45] | MATCH |
| Q5 | [not found] | [not found] | [] | MATCH |

**Result: 5/5 questions answered correctly**

---

## Multiple Testing & Verification (`multi_test.py`)

To guarantee strict compliance and eliminate any risk of hallucination or brittle matching, a comprehensive test suite with 18 automated tests is provided:

```bash
python3 multi_test.py
```

### Test Coverage:
1. **Primary Assessment Suite**: Exact verification of Q1, Q2, Q3, Q4, Q5 against expected answers and sources.
2. **Semantic Paraphrase Suite**: Verifies that rephrasing questions still extracts the grounded facts and cites the exact source page.
3. **Negative & Guardrail Suite**: Tests out-of-domain queries (Sam Altman, Geography, Barbie movie, Tech CEOs, Physics) and confirms 100% return `[not found]` with empty sources `[]`.
4. **Schema Compliance**: Validates that all responses adhere strictly to `{"answer": <str>, "sources": [<page_num>, ...]}`.
5. **Persistence & Latency**: Verifies that subsequent queries complete in under 1 second without re-embedding.

---

## Vercel Deployment

This project includes a serverless web deployment configuration for Vercel:
- **Live URL**: [https://pravin-project-7.vercel.app](https://pravin-project-7.vercel.app)
- **API Endpoint**: `https://pravin-project-7.vercel.app/api/answer?q=<question>`
- `api/index.py`: Serverless RAG answering endpoint (`/api/answer?q=...`)
- `public/index.html`: Interactive web dashboard (uses pure SVG icons, zero emojis)
- `vercel.json`: Routing and configuration for serverless Python runtime

### Deploying to Vercel via CLI:
```bash
# Using a Vercel personal access token
vercel deploy --prod --token <YOUR_VERCEL_TOKEN> --yes
```

Or connect the GitHub repository directly at [vercel.com](https://vercel.com/new).

---

## Execution Screenshot

*(See the terminal output from running `python3 main.py`)*

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `chroma_db` | ChromaDB storage directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name |
| `PDF_PATH` | `data/transformer_rag_research_paper_50p.pdf` | PDF file path |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

---

## Key Features

- **Strict Grounding**: Answers come ONLY from the PDF — no external knowledge, no hallucination
- **Not Found Handling**: Returns exactly `[not found]` when information is absent from the PDF
- **Source References**: Every answer includes the PDF page number(s) where the information was found
- **Database Persistence**: ChromaDB database persists to disk — no re-embedding on restart
- **Local Embeddings**: No API keys needed — the sentence-transformers model runs locally on CPU
