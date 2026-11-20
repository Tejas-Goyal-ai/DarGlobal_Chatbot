import asyncio
import json
import re
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

SEED_URLS = [
    # DarGlobal main & project hubs
    {"site": "DarGlobal", "url": "https://darglobal.co.uk"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/projects"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/projects/properties-in-saudi-arabia"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/projects/properties-in-oman"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/the-astera"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/partners/pagani"},
    {"site": "DarGlobal", "url": "https://darglobal.co.uk/partners/elie-saab"},
    # Wasalt English hubs & subdomains
    {"site": "Wasalt", "url": "https://wasalt.sa/en"},
    {"site": "Wasalt", "url": "https://wasalt.sa/en/corporate"},
    {"site": "Wasalt", "url": "https://wasalt.sa/en/wb-daily-rentals"},
    {"site": "Wasalt", "url": "https://auction.wasalt.sa/en"},
    {"site": "Wasalt", "url": "https://business.wasalt.sa/en/login"},
]

ALLOWED_DOMAINS = {
    "DarGlobal": ["darglobal.co.uk"],
    "Wasalt": ["wasalt.sa", "auction.wasalt.sa", "business.wasalt.sa"],
}

MAX_PAGES_PER_SITE = 80  # Captures all core projects, listings, and info pages cleanly


def clean_html_text(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    # Strip noise elements
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "cookie"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, meta_desc, text


def is_valid_url(url: str, site: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.netloc.lower().replace("www.", "")
    if host.startswith("preprod."):
        return False
    if not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS[site]):
        return False
    path = parsed.path.lower()
    # Skip binary assets and non-English duplicates so RAG stays focused and clean
    if any(path.endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".webp", ".svg", ".zip", ".xml")):
        return False
    if "/ar/" in path or path.endswith("/ar") or "/zh/" in path:
        return False
    return True


async def fetch_sitemap_links(site: str, sitemap_url: str) -> list[str]:
    found = []
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(sitemap_url)
            if resp.status_code == 200 and "<loc>" in resp.text:
                soup = BeautifulSoup(resp.text, "xml")
                for loc in soup.find_all("loc"):
                    u = loc.text.strip().rstrip("/")
                    if is_valid_url(u, site):
                        found.append(u)
    except Exception:
        pass
    return list(dict.fromkeys(found))


async def crawl():
    queues = {"DarGlobal": [], "Wasalt": []}

    # 1. Try sitemaps first
    for u in await fetch_sitemap_links("DarGlobal", "https://darglobal.co.uk/sitemap.xml"):
        queues["DarGlobal"].append(u)
    for u in await fetch_sitemap_links("Wasalt", "https://wasalt.sa/sitemap.xml"):
        queues["Wasalt"].append(u)

    # 2. Add curated high-value seed URLs
    for item in SEED_URLS:
        queues[item["site"]].append(item["url"].rstrip("/"))

    for site in queues:
        queues[site] = list(dict.fromkeys(queues[site]))

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        for site in ["DarGlobal", "Wasalt"]:
            visited = set()
            site_count = 0

            while queues[site] and site_count < MAX_PAGES_PER_SITE:
                url = queues[site].pop(0)
                if url in visited:
                    continue
                visited.add(url)

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                    await page.wait_for_timeout(2000)  # Let JS/React property cards hydrate

                    final_url = page.url.split("#")[0].rstrip("/")
                    html = await page.content()
                    title, meta_desc, body_text = clean_html_text(html)

                    if len(body_text) < 200:
                        continue

                    # Extract internal links for deep crawling
                    hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.getAttribute('href'))")
                    for href in hrefs:
                        if not href:
                            continue
                        full_u = urljoin(final_url, href).split("#")[0].split("?")[0].rstrip("/")
                        if full_u not in visited and is_valid_url(full_u, site):
                            queues[site].append(full_u)

                    results.append({
                        "site": site,
                        "url": final_url,
                        "title": title or final_url,
                        "description": meta_desc,
                        "content": body_text[:15000],
                    })
                    site_count += 1
                    print(f"[{site} {site_count}/{MAX_PAGES_PER_SITE}] {title[:55]} | {final_url}")

                except Exception as e:
                    print(f"[{site}] Failed {url}: {e}")

        await browser.close()

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    dg_total = sum(1 for r in results if r["site"] == "DarGlobal")
    ws_total = sum(1 for r in results if r["site"] == "Wasalt")
    print(f"\nDONE! Saved {len(results)} pages total (DarGlobal: {dg_total}, Wasalt: {ws_total}) to data.json")


if __name__ == "__main__":
    asyncio.run(crawl())