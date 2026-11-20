# DarGlobal & Wasalt AI Property Concierge

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built to answer property, investment, and project inquiries using publicly scraped data from **DarGlobal** and **Wasalt**.

**Live Demo:** [https://darglobal-chatbot-zcby.onrender.com](https://darglobal-chatbot-zcby.onrender.com)  
*(Note: Hosted on Render's free tier — initial requests after inactivity may take ~30–40 seconds to spin up.)*

---

## Architecture & Tech Stack

- **Data Ingestion & Cleaning:** Custom Python scrapers (`scrapper.py`, `clean_data.py`) extracting and normalizing property listings, project details, and metadata into `data.json`.
- **Embeddings & Vector Store:** `FastEmbed` (`BAAI/bge-small-en-v1.5`) paired with persistent `ChromaDB` for low-latency semantic retrieval.
- **Backend & Memory:** `FastAPI` serving real-time Server-Sent Events (SSE) token streaming and `SQLite` (`chat_memory.db`) for multi-turn session history.
- **LLM Inference:** Integrated with **OpenRouter** (free tier models) with grounded context injection and clickable source citations.
- **Containerization & Hosting:** Fully containerized via `Dockerfile` (pre-baking embeddings and vector indexes during image build) and deployed on **Render**.

---

## Project Structure

```text
├── app.py              # FastAPI server, RAG retrieval, OpenRouter streaming & UI
├── ingest.py           # Generates embeddings and populates ChromaDB from data.json
├── scrapper.py         # Web scraper for DarGlobal and Wasalt property listings
├── clean_data.py       # Data normalization and deduplication pipeline
├── data.json           # Cleaned structured property dataset
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-step container build with pre-indexed ChromaDB
└── .dockerignore       # Excludes local virtual environments, caches, and secrets
