# Smart RSS Crypto News Chatbot

A CLI chatbot that:
- pulls cryptocurrency news from RSS feeds into a local SQLite database,
- builds a local semantic-search index (Chroma) over recent articles using **BAAI/bge-m3** embeddings,
- chats over your news + can optionally use live web tools (DuckDuckGo search, CoinGecko prices, URL text extraction).

## Project structure

- `chat.py` — main CLI app. Loads cached articles, maintains conversation history, and streams tokens to stdout.
- `fetcher_service.py` — background RSS fetcher (runs every ~30 min by default).
- `getRSS.py` — RSS feed list + fetching + date filtering helpers.
- `database.py` — SQLite article store with duplicate protection.
- `vector_store.py` — Chroma vector store manager using **BAAI/bge-m3** embeddings.
- `tools.py` — LangChain tools (web search, crypto price via CoinGecko, fetch webpage content, etc.).
- `setup_mocking.py` / `create_mock_data.py` — build a mock DB + mock vector store for testing.
- `test_database.py` — interactive DB viewer + basic performance checks.

## Requirements

- Python 3.10+ recommended
- Local model used for this project: Qwen-32B
- In chat.py, LLM is currently set as the small LLM, **Ollama** installed + running locally on my laptop - NOT recommeneded for this program as the LLM is too basic, if you have an openai api key you could edit the script to use that.

### Python dependencies (core)

- `langchain`, `langchain-core`, `langchain-community`
- `langchain-ollama`
- `langchain-chroma` + `chromadb`
- `langchain-huggingface` (pulls HF embedding model **BAAI/bge-m3** on first use)
- `feedparser`, `python-dateutil`, `requests`
- `schedule`
- `ddgs` (DuckDuckGo search)
- `beautifulsoup4`
- `python-dotenv`
- `sqlalchemy` (used by the app runtime)

Quick install (example):

```bash
pip install -U langchain langchain-core langchain-community langchain-ollama langchain-chroma chromadb langchain-huggingface
pip install -U feedparser python-dateutil requests schedule ddgs beautifulsoup4 python-dotenv sqlalchemy
```

## Setup

1) **Start Ollama** and pull the default model:

```bash
ollama pull llama3.2
```

2) (Optional) Create a `.env` file if you later add keys or config; the project already loads env vars.

## Run

### Chatbot (main)

```bash
python chat.py
```

On startup, it loads the most recent cached articles (default `days_back = 7`) and optionally starts the periodic fetcher.

### Background fetcher (standalone)

If you want the fetcher as a separate process:

```bash
python fetcher_service.py
```

It fetches RSS feeds and stores new articles into the database on a schedule (default 30 minutes).

## What it can do

Inside the chat, you can ask about:
- “latest crypto news”
- “what happened with <topic> recently”
- “what’s the current bitcoin price”
- “search the web for SEC crypto regulations”
For example.


The help screen lists commands like `help`, `stats`, and `status`.

## Mocking / testing

To generate a clean mock environment (mock DB + mock vector store), run:

```bash
python setup_mocking.py
```

This deletes any previous mock artifacts (`mocking_articles.db`, `mock_vector_store`) and creates fresh mock articles for testing.

Then try:

```bash
python test_database.py
```

## Notes

- Embeddings are computed with **BAAI/bge-m3** and stored in a persistent Chroma directory (default `./vector_store`).
- Live tools include CoinGecko price lookup and DuckDuckGo search.




mock articles stored in database "mocking_articles.db"

use create_mock_data.py to generate 3 mock articles for each day for the last 2 weeks (also deletes any previous mock data generated)
