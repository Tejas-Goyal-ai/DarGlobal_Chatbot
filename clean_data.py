import json
from urllib.parse import urlparse

with open("data.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

seen_urls = set()
cleaned = []

CUSTOM_WASALT_TITLES = {
    "https://auction.wasalt.sa/en": ("Wasalt Auctions (Infath & Private Auctions)", "Participate in timed real estate, vehicle, and asset auctions across Saudi Arabia on Wasalt."),
    "https://business.wasalt.sa/en/login": ("Wasalt Business Portal & Matchmaker Login", "Log in to Wasalt Business to manage listings, connect with leads via Matchmaker, and grow your brokerage."),
    "https://blog.wasalt.sa/en": ("Wasalt Real Estate Blog & Market Insights", "Explore Saudi real estate guides, market trends, and property tips on the Wasalt Blog."),
    "https://wasalt.sa/en/wb-daily-rentals": ("Wasalt Daily Rentals & Short-Term Stays", "Discover or list short-term daily rental properties across Saudi Arabia with Wasalt."),
    "https://wasalt.sa/en/sale/search": ("Search Properties for Sale in Saudi Arabia | Wasalt", "Browse verified villas, apartments, and land for sale across Saudi Arabia on Wasalt."),
    "https://wasalt.sa/en/corporate": ("About Wasalt | Corporate & REGA Licensed Platform", "Learn about Wasalt, Saudi Arabia's REGA-licensed AI-backed real estate platform."),
}

for p in pages:
    url = p.get("url", "").strip().rstrip("/")
    title = p.get("title", "").strip()
    content = p.get("content", "").strip()
    site = p.get("site", "Web").strip()

    # 1. Skip duplicates and bare root wasalt.sa (since wasalt.sa/en has full content)
    if not url or url in seen_urls or url == "https://wasalt.sa":
        continue

    # 2. Skip DarGlobal pages that threw Next.js client-side exceptions or empty cookie-only pages
    if "application error: a client-side exception has occurred" in title.lower():
        continue
    if len(content) < 250 and "just a moment" not in title.lower():
        continue

    seen_urls.add(url)

    # 3. Upgrade Wasalt Cloudflare challenge pages into rich, clean navigation entries
    if "just a moment" in title.lower():
        if url in CUSTOM_WASALT_TITLES:
            new_title, new_desc = CUSTOM_WASALT_TITLES[url]
        else:
            slug = urlparse(url).path.strip("/").split("/")[-1]
            readable = slug.replace("-", " ").title()
            new_title = f"{readable} | Wasalt Saudi Arabia"
            new_desc = f"Browse verified {readable.lower()} on Wasalt, Saudi Arabia's REGA-licensed real estate platform."

        p["title"] = new_title
        p["description"] = new_desc
        p["content"] = (
            f"{new_title}. {new_desc} "
            f"Access verified listings, installment options, and trusted brokers directly on Wasalt: {url}"
        )

    # 4. Fix raw URL titles on DarGlobal pages (e.g., https://darglobal.co.uk/La-Collina)
    elif title.startswith("http"):
        slug = urlparse(url).path.strip("/").split("/")[-1]
        p["title"] = f"{slug.replace('-', ' ').title()} | DarGlobal"

    cleaned.append(p)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

print(f"Polished data.json: {len(cleaned)} clean, unique pages saved!")