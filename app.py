import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from openrouter import OpenRouter
from pydantic import BaseModel

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "real_estate_kb"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DB_PATH = "./chat_memory.db"

# Speed-ordered free models (low-latency active parameters first, auto-router fallback last)
FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-3-1.7b:free",
    "openrouter/free",
]

app = FastAPI(title="DarGlobal & Wasalt AI Property Concierge")

# Load pre-saved ChromaDB & lightweight FastEmbed query encoder
embedder = TextEmbedding(model_name=EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    source_filter: Optional[str] = "All"


def get_history(session_id: str, limit: int = 8) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def save_message(session_id: str, role: str, content: str, sources_json: str = "[]"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, sources_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, sources_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


@app.get("/api/history/{session_id}")
def fetch_session_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, sources_json FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "sources": json.loads(r[2] or "[]")} for r in rows]


@app.delete("/api/history/{session_id}")
def clear_session_history(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "cleared"}


@app.post("/api/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set.")

    history = get_history(req.session_id, limit=8)

    contextual_query = req.message
    if history:
        recent_turns = " | ".join([m["content"] for m in history[-2:] if m["role"] == "user"])
        if recent_turns:
            contextual_query = f"{recent_turns} -> {req.message}"

    query_vector = list(embedder.embed([contextual_query]))[0].tolist()

    where_filter = {"site": req.source_filter} if req.source_filter in ("DarGlobal", "Wasalt") else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=4,
        where=where_filter,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    unique_sources = []
    seen_urls = set()
    context_blocks = []

    for doc, meta in zip(docs, metas):
        context_blocks.append(doc[:1200])
        url = meta.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append({
                "site": meta.get("site", "Web"),
                "title": meta.get("title", url),
                "url": url,
                "description": meta.get("description", ""),
            })

    system_prompt = (
        "You are an expert AI Property & Navigation Concierge for DarGlobal and Wasalt.\n"
        "1. Answer strictly using the retrieved website context and conversation history.\n"
        "2. Remember user preferences from earlier in this session.\n"
        "3. Provide exact clickable Markdown links [Page Title](URL) whenever referencing projects, services, or navigation requests.\n"
        "4. Be concise, structured with bullet points, and never invent URLs or prices.\n\n"
        "CONTEXT:\n" + "\n---\n".join(context_blocks)
    )

    messages_payload = [{"role": "system", "content": system_prompt}]
    messages_payload.extend(history)
    messages_payload.append({"role": "user", "content": req.message})

    def event_generator():
        full_reply = ""
        try:
            with OpenRouter(api_key=OPENROUTER_API_KEY) as or_client:
                # OpenRouter allows at most 3 models in the 'models' parameter.
                # Try speed-ordered models with fallback routing.
                models_to_try = [
                    (FREE_MODELS[0], FREE_MODELS[:3]),
                    (FREE_MODELS[1], FREE_MODELS[1:4]),
                    (FREE_MODELS[3], None),
                ]
                for primary, fallback_list in models_to_try:
                    try:
                        kwargs = {
                            "model": primary,
                            "messages": messages_payload,
                            "stream": True,
                            "max_tokens": 600,
                            "temperature": 0.3,
                        }
                        if fallback_list:
                            kwargs["models"] = fallback_list

                        stream = or_client.chat.send(**kwargs)
                        with stream as event_stream:
                            for chunk in event_stream:
                                if not chunk.choices:
                                    continue
                                delta = getattr(chunk.choices[0].delta, "content", None) or ""
                                if delta:
                                    full_reply += delta
                                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
                        if full_reply:
                            break
                    except Exception as e:
                        print(f"OpenRouter stream error ({primary}):", repr(e))
                        continue
        except Exception as e:
            print("OpenRouter client error:", repr(e))

        if not full_reply:
            fallback_msg = (
                "I encountered a temporary rate limit on the free AI model. "
                "Please retry in a moment or click any of the verified source cards below."
            )
            full_reply = fallback_msg
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"

        # Deliver clickable navigation cards after text is generated
        if unique_sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': unique_sources})}\n\n"

        save_message(req.session_id, "user", req.message)
        save_message(req.session_id, "assistant", full_reply, json.dumps(unique_sources))
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DarGlobal & Wasalt — AI Property & Navigation Concierge</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #F7F3EC;
      --card: #FDFBF7;
      --bubble: #EFE8DC;
      --border: #DFD4C3;
      --text: #2C221E;
      --muted: #6E5D53;
      --accent: #9A6B32;
      --accent-light: #B58446;
      --serif: 'Cormorant Garamond', Georgia, 'Times New Roman', serif;
      --sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      color: var(--text);
      font-family: var(--sans);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      background:
        radial-gradient(1100px 620px at 16% -10%, rgba(181, 132, 70, 0.10), transparent 60%),
        radial-gradient(920px 560px at 94% 6%, rgba(154, 107, 50, 0.07), transparent 55%),
        var(--bg);
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 9999;
      opacity: 0.035;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E");
    }
    header {
      padding: 18px 32px;
      background: rgba(253, 251, 247, 0.92);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
      position: relative;
      z-index: 10;
    }
    .eyebrow {
      font-family: var(--sans);
      font-size: 0.63rem;
      font-weight: 700;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 3px;
    }
    header h1 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 700;
      font-size: 1.65rem;
      line-height: 1.1;
      letter-spacing: 0.2px;
      color: var(--text);
    }
    header h1 .amp { color: var(--accent); font-style: italic; }
    header h1 .title-sub { font-size: 0.58em; font-weight: 600; font-style: italic; color: var(--muted); }
    .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    select, button.secondary {
      font-family: var(--sans);
      background: var(--card);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 9px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 0.8rem;
      transition: border-color 0.2s, color 0.2s, box-shadow 0.2s;
    }
    select:hover, button.secondary:hover { border-color: var(--accent-light); color: var(--text); box-shadow: 0 3px 10px rgba(92, 62, 36, 0.08); }
    button.primary {
      font-family: var(--sans);
      background: linear-gradient(135deg, #9A6B32, #B58446);
      color: #FDFBF7;
      font-weight: 700;
      border: none;
      padding: 11px 26px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 0.86rem;
      letter-spacing: 0.03em;
      box-shadow: 0 4px 14px rgba(154, 107, 50, 0.25);
      transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
    }
    button.primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(154, 107, 50, 0.35); }
    button.primary:disabled { opacity: 0.72; cursor: not-allowed; }
    button.primary.loading { animation: btnPulse 1.3s ease-in-out infinite; }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 32px 24px 48px;
      max-width: 880px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
      scrollbar-width: thin;
      scrollbar-color: #C9B9A3 transparent;
    }
    #chat::-webkit-scrollbar { width: 8px; }
    #chat::-webkit-scrollbar-track { background: transparent; }
    #chat::-webkit-scrollbar-thumb { background: #C9B9A3; border-radius: 999px; border: 2px solid var(--bg); }
    #chat::-webkit-scrollbar-thumb:hover { background: var(--accent); }
    .msg {
      font-family: var(--sans);
      padding: 16px 20px;
      border-radius: 16px;
      line-height: 1.65;
      font-size: 0.94rem;
      max-width: 88%;
      word-break: break-word;
      animation: fadeSlideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .msg.user {
      background: var(--bubble);
      align-self: flex-end;
      border: 1px solid #E2D7C6;
      border-bottom-right-radius: 5px;
      color: var(--text);
      box-shadow: 0 2px 8px rgba(44, 34, 30, 0.04);
    }
    .msg.assistant {
      background: var(--card);
      align-self: flex-start;
      border: 1px solid var(--border);
      width: 100%;
      max-width: 100%;
      border-top-left-radius: 5px;
      box-shadow: 0 6px 22px rgba(44, 34, 30, 0.05);
    }
    .assistant p { margin: 0.5em 0; }
    .assistant ul, .assistant ol { margin: 0.5em 0; padding-left: 1.25rem; }
    .assistant li { margin: 0.25em 0; }
    .assistant h1, .assistant h2, .assistant h3, .assistant h4 {
      font-family: var(--serif);
      font-weight: 700;
      color: var(--text);
      line-height: 1.2;
      margin: 0.7em 0 0.35em;
    }
    .assistant h1 { font-size: 1.7rem; }
    .assistant h2 { font-size: 1.45rem; }
    .assistant h3 { font-size: 1.22rem; }
    .assistant h4 { font-size: 1.08rem; }
    .assistant a { color: var(--accent); text-decoration: underline; text-decoration-color: rgba(154, 107, 50, 0.35); text-underline-offset: 3px; }
    .assistant a:hover { color: var(--accent-light); text-decoration-color: var(--accent-light); }
    .assistant strong { color: var(--text); }
    .assistant code { background: #F1E9DC; padding: 1px 5px; border-radius: 5px; font-size: 0.88em; }
    .assistant blockquote { margin: 0.6em 0; padding-left: 14px; border-left: 3px solid var(--accent-light); color: var(--muted); font-style: italic; }
    .thinking { display: flex; align-items: center; gap: 12px; padding: 4px 2px; }
    .thinking-dots { display: inline-flex; gap: 5px; }
    .thinking-dots span {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: dotPulse 1.1s ease-in-out infinite;
    }
    .thinking-dots span:nth-child(2) { background: var(--accent-light); animation-delay: 0.15s; }
    .thinking-dots span:nth-child(3) { background: var(--text); animation-delay: 0.3s; }
    .thinking-label {
      font-family: var(--serif);
      font-style: italic;
      font-size: 1.05rem;
      background: linear-gradient(90deg, var(--muted) 25%, var(--accent-light) 50%, var(--muted) 75%);
      background-size: 200% 100%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: shimmer 1.9s linear infinite;
    }
    .stream-cursor {
      display: inline-block;
      width: 2px;
      height: 1.05em;
      margin-left: 2px;
      vertical-align: -0.16em;
      background: var(--accent);
      border-radius: 1px;
      animation: cursorBlink 1s steps(1) infinite;
    }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 14px; }
    .cards:empty { display: none; }
    .source-card {
      display: block;
      text-decoration: none;
      background: var(--card);
      border: 1px solid var(--border);
      padding: 14px 16px;
      border-radius: 12px;
      animation: fadeIn 0.45s ease backwards;
      transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .source-card:hover {
      transform: translateY(-3px);
      border-color: var(--accent-light);
      box-shadow: 0 10px 24px rgba(92, 62, 36, 0.14);
    }
    .badge { font-family: var(--sans); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
    .card-title { font-family: var(--serif); font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 5px 0 6px; line-height: 1.2; }
    .card-desc { font-family: var(--sans); font-size: 0.76rem; color: var(--muted); line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .starters { max-width: 880px; width: 100%; margin: 0 auto; padding: 0 24px 8px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .starters-label { flex-basis: 100%; font-family: var(--sans); font-size: 0.62rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); }
    .chip {
      font-family: var(--sans);
      font-size: 0.78rem;
      background: rgba(253, 251, 247, 0.9);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 8px 15px;
      border-radius: 999px;
      cursor: pointer;
      transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease, color 0.22s ease, background 0.22s ease;
    }
    .chip:hover { transform: translateY(-2px); color: var(--accent); border-color: var(--accent-light); background: var(--card); box-shadow: 0 6px 16px rgba(92, 62, 36, 0.10); }
    footer { background: transparent; border-top: 1px solid var(--border); padding: 14px 24px 20px; position: relative; z-index: 10; }
    .input-wrap {
      max-width: 880px;
      margin: 0 auto;
      display: flex;
      gap: 10px;
      align-items: center;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 8px;
      box-shadow: 0 8px 30px rgba(44, 34, 30, 0.08);
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .input-wrap:focus-within { border-color: var(--accent-light); box-shadow: 0 8px 30px rgba(154, 107, 50, 0.14); }
    input { flex: 1; min-width: 0; background: transparent; border: none; color: var(--text); padding: 10px 14px; font-size: 0.94rem; font-family: var(--sans); outline: none; }
    input::placeholder { color: #9C8B7E; }
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes dotPulse { 0%, 60%, 100% { transform: translateY(0); opacity: 0.45; } 30% { transform: translateY(-4px); opacity: 1; } }
    @keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
    @keyframes cursorBlink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
    @keyframes btnPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(181, 132, 70, 0.45); } 50% { box-shadow: 0 0 0 7px rgba(181, 132, 70, 0); } }
    @media (max-width: 640px) {
      header { padding: 14px 18px; }
      header h1 { font-size: 1.35rem; }
      #chat { padding: 22px 16px 36px; }
      .starters { padding: 0 16px 8px; }
      footer { padding: 12px 16px 16px; }
      .msg { max-width: 94%; }
    }
    @media (prefers-reduced-motion: reduce) {
      .msg, .source-card, button.primary, .thinking-dots span, .thinking-label, .stream-cursor { animation: none !important; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="eyebrow">Private Real Estate Intelligence</div>
      <h1>DarGlobal <span class="amp">&</span> Wasalt <span class="title-sub">— AI Property & Navigation Concierge</span></h1>
    </div>
    <div class="controls">
      <select id="sourceFilter">
        <option value="All">All Sources (DarGlobal + Wasalt)</option>
        <option value="DarGlobal">DarGlobal Only</option>
        <option value="Wasalt">Wasalt Only</option>
      </select>
      <button id="resetBtn" class="secondary">Reset Session Memory</button>
    </div>
  </header>

  <div id="chat"></div>

  <div class="starters">
    <span class="starters-label">Suggested inquiries</span>
    <button class="chip" data-preset="Show me DarGlobal luxury projects and provide direct links to explore them.">Explore DarGlobal Projects</button>
    <button class="chip" data-preset="What properties, auctions, and services does Wasalt offer? Give me direct links.">Explore Wasalt Services</button>
    <button class="chip" data-preset="Compare DarGlobal and Wasalt in terms of their offerings and target audience.">Compare DarGlobal vs Wasalt</button>
  </div>

  <footer>
    <div class="input-wrap">
      <input id="prompt" placeholder="Ask about projects, listings, or ask me to navigate to any DarGlobal / Wasalt page..." />
      <button id="sendBtn" class="primary">Send</button>
    </div>
  </footer>

  <script>
    let sessionId = localStorage.getItem('dg_session_id');
    if (!sessionId) {
      sessionId = 'sess_' + Math.random().toString(36).substring(2, 12);
      localStorage.setItem('dg_session_id', sessionId);
    }

    const chatEl = document.getElementById('chat');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    let isStreaming = false;

    function safeRenderMarkdown(content) {
      if (typeof marked !== 'undefined' && marked.parse) {
        return marked.parse(content);
      }
      return content;
    }

    function attachCursor(root, cursor) {
      const candidates = root.querySelectorAll('p, li, h1, h2, h3, h4, blockquote, strong, em, a');
      const host = candidates.length ? candidates[candidates.length - 1] : root;
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
      host.appendChild(cursor);
    }

    function renderMessage(role, content, sources = []) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.innerHTML = role === 'assistant' ? safeRenderMarkdown(content) : content;

      if (role === 'assistant' && sources && sources.length > 0) {
        const cards = document.createElement('div');
        cards.className = 'cards';
        sources.forEach((s, index) => {
          const a = document.createElement('a');
          a.className = 'source-card';
          a.style.animationDelay = (index * 0.08) + 's';
          a.href = s.url;
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          a.innerHTML = `<div class="badge">${s.site} ↗</div><div class="card-title">${s.title}</div><div class="card-desc">${s.description || s.url}</div>`;
          cards.appendChild(a);
        });
        div.appendChild(cards);
      }
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    async function loadHistory() {
      try {
        const res = await fetch('/api/history/' + sessionId);
        const data = await res.json();
        if (!data || data.length === 0) {
          renderMessage('assistant', 'Welcome! I have indexed **DarGlobal** and **Wasalt**. Ask me about specific luxury developments, property services, or ask me to navigate to any page—I will remember your preferences across our session and provide direct clickable links.');
        } else {
          data.forEach(m => renderMessage(m.role, m.content, m.sources));
        }
      } catch (err) {
        console.error('Failed to load history:', err);
        renderMessage('assistant', 'Welcome! How can I assist you with DarGlobal or Wasalt today?');
      }
    }

    function sendPreset(text) {
      if (promptInput) {
        promptInput.value = text;
        sendMessage();
      }
    }

    async function sendMessage() {
      if (isStreaming) {
        console.log('SendMessage ignored: already streaming.');
        return;
      }
      const text = promptInput ? promptInput.value.trim() : '';
      if (!text) return;

      console.log('User prompt submitted:', text);

      // Disable Send button & update UI immediately
      if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerText = 'Sending...';
        sendBtn.classList.add('loading');
      }
      promptInput.value = '';
      isStreaming = true;

      // Immediately render user bubble
      renderMessage('user', text);

      // Assistant streaming placeholder with elegant thinking indicator
      const msgDiv = document.createElement('div');
      msgDiv.className = 'msg assistant';
      const textDiv = document.createElement('div');
      textDiv.className = 'stream-text';
      textDiv.innerHTML =
        '<div class="thinking">' +
          '<span class="thinking-dots"><span></span><span></span><span></span></span>' +
          '<span class="thinking-label">Curating portfolio insights...</span>' +
        '</div>';
      const cardsDiv = document.createElement('div');
      cardsDiv.className = 'cards';
      msgDiv.appendChild(textDiv);
      msgDiv.appendChild(cardsDiv);
      chatEl.appendChild(msgDiv);
      chatEl.scrollTop = chatEl.scrollHeight;

      const cursor = document.createElement('span');
      cursor.className = 'stream-cursor';

      let accumulated = '';
      let pendingSources = [];

      function renderSourceCards(sources) {
        cardsDiv.innerHTML = '';
        sources.forEach((s, index) => {
          const a = document.createElement('a');
          a.className = 'source-card';
          a.style.animationDelay = (index * 0.08) + 's';
          a.href = s.url;
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          a.innerHTML = `<div class="badge">${s.site} ↗</div><div class="card-title">${s.title}</div><div class="card-desc">${s.description || s.url}</div>`;
          cardsDiv.appendChild(a);
        });
        chatEl.scrollTop = chatEl.scrollHeight;
      }

      try {
        console.log('Initiating stream fetch to /api/chat/stream...');
        const res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: text,
            source_filter: document.getElementById('sourceFilter').value
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          const errMsg = errData.detail || ('HTTP ' + res.status);
          console.error('Stream request failed:', errMsg);
          textDiv.innerText = 'Error: ' + errMsg;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const payload = JSON.parse(line.slice(6));

              if (payload.type === 'sources') {
                console.log('Received sources:', payload.sources);
                pendingSources = payload.sources;
                if (accumulated) {
                  renderSourceCards(pendingSources);
                }
              } else if (payload.type === 'token') {
                accumulated += payload.content;
                textDiv.innerHTML = safeRenderMarkdown(accumulated);
                attachCursor(textDiv, cursor);
                chatEl.scrollTop = chatEl.scrollHeight;
              } else if (payload.type === 'done') {
                console.log('Stream completed successfully.');
                if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
                if (pendingSources.length > 0 && cardsDiv.children.length === 0) {
                  renderSourceCards(pendingSources);
                }
              }
            } catch (jsonErr) {
              console.warn('Failed to parse SSE payload:', line, jsonErr);
            }
          }
        }
      } catch (err) {
        console.error('Fetch stream error:', err);
        textDiv.innerText = 'Network error: ' + err.message;
      } finally {
        if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
        if (pendingSources.length > 0 && cardsDiv.children.length === 0) {
          renderSourceCards(pendingSources);
        }
        isStreaming = false;
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.innerText = 'Send';
          sendBtn.classList.remove('loading');
        }
        if (promptInput) {
          promptInput.focus();
        }
        console.log('Stream finished. Send button re-enabled.');
      }
    }

    async function clearMemory() {
      try {
        await fetch('/api/history/' + sessionId, { method: 'DELETE' });
        console.log('Session memory cleared.');
      } catch (e) {
        console.error('Failed to clear memory:', e);
      }
      chatEl.innerHTML = '';
      loadHistory();
    }

    // Attach proper DOM Event Listeners
    if (sendBtn) {
      sendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        sendMessage();
      });
    }

    if (promptInput) {
      promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          sendMessage();
        }
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', (e) => {
        e.preventDefault();
        clearMemory();
      });
    }

    document.querySelectorAll('.chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const preset = btn.getAttribute('data-preset');
        if (preset) sendPreset(preset);
      });
    });

    loadHistory();
  </script>
</body>
</html>
"""