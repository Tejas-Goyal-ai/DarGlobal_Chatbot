import json
import chromadb
from fastembed import TextEmbedding

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "real_estate_kb"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def main():
    with open("data.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"Loaded {len(pages)} pages. Initializing {EMBED_MODEL}...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    seen_urls = set()
    idx = 0

    for page in pages:
        url = page.get("url", "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        site = page.get("site", "Web")
        title = page.get("title", url).strip()
        desc = page.get("description", "").strip()
        content = page.get("content", "").strip()

        header = f"Source: {site} | Title: {title} | URL: {url}\nSummary: {desc}\nContent: "
        for chunk in chunk_text(content):
            ids.append(f"chunk_{idx}")
            docs.append(header + chunk)
            metas.append({
                "site": site,
                "title": title[:250],
                "url": url,
                "description": desc[:350],
            })
            idx += 1

    print(f"Generating embeddings for {len(docs)} chunks (one-time process)...")
    embeddings = [vec.tolist() for vec in embedder.embed(docs, batch_size=64)]

    batch_size = 250
    for i in range(0, len(docs), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
        )

    print(f"DONE! Saved {len(docs)} chunks across {len(seen_urls)} unique pages into {CHROMA_DIR}/")


if __name__ == "__main__":
    main()