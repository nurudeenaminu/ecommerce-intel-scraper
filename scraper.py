import asyncio
import json
import random
import time
from pathlib import Path
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

# ─── CONFIG ───────────────────────────────────────────────
SEARCH_QUERIES = [
    "wireless headphones",
    "bluetooth speakers",
    "wireless earbuds",
    "noise cancelling headphones",
    "gaming headsets",
    "smart watches",
    "wireless keyboards",
    "webcams for streaming",
    "portable chargers",
    "laptop stands",
]
MAX_PAGES    = 5   # 5 pages × 10 categories × ~8 products = ~400 products
OUTPUT_FILE  = "data/raw/products_raw.json"
MAX_RETRIES  = 3   # retry logic for failed pages

# ─── USER-AGENT ROTATION ──────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ─── PROXY STUB ───────────────────────────────────────────
# To enable proxies: set USE_PROXY = True and fill PROXY_LIST
# Format: "http://user:pass@host:port" or "http://host:port"
USE_PROXY  = False
PROXY_LIST = [
    # "http://proxy1:port",
    # "http://proxy2:port",
]

def get_proxy():
    if USE_PROXY and PROXY_LIST:
        return {"server": random.choice(PROXY_LIST)}
    return None

# ─── HELPERS ──────────────────────────────────────────────
async def safe_get(page, selector, script):
    try:
        el = await page.query_selector(selector)
        if el:
            return await page.eval_on_selector(selector, script)
    except Exception:
        pass
    return None

async def safe_get_all(page, selector, script):
    try:
        els = await page.query_selector_all(selector)
        if els:
            return await page.eval_on_selector_all(selector, script)
    except Exception:
        pass
    return []

async def get_product_links(page):
    """Multi-strategy product link extraction."""
    # Strategy 1: data-asin (most reliable)
    links = await page.evaluate("""
        () => {
            const items = document.querySelectorAll('[data-asin]');
            const hrefs = [];
            items.forEach(item => {
                const asin = item.getAttribute('data-asin');
                if (asin && asin.length > 0) {
                    const a = item.querySelector('a[href*="/dp/"]');
                    if (a) hrefs.push(a.href);
                }
            });
            return [...new Set(hrefs)];
        }
    """)
    if links:
        return links

    # Strategy 2: any /dp/ anchor (broad fallback)
    links = await page.evaluate("""
        () => {
            const all = document.querySelectorAll('a[href*="/dp/"]');
            return [...new Set(
                Array.from(all)
                    .map(a => a.href)
                    .filter(h => h.includes('amazon.com') && h.includes('/dp/'))
            )];
        }
    """)
    return links

async def goto_with_retry(page, url, retries=MAX_RETRIES):
    """Exponential backoff retry for failed page loads."""
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return True
        except Exception as e:
            wait = 2 ** attempt   # 1s, 2s, 4s
            print(f"     ⚠ Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
    print(f"     ✗ All {retries} attempts failed for {url[:60]}")
    return False

def save_progress(products):
    """Save after every product — never lose data."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

# ─── MAIN SCRAPER ─────────────────────────────────────────
async def scrape_amazon():
    products     = []
    seen_urls    = set()   # avoid duplicate product pages
    failed_pages = []      # log all failures for the summary

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=80,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

        for query in SEARCH_QUERIES:
            print(f"\n{'='*58}")
            print(f"  CATEGORY: {query.upper()}")
            print(f"{'='*58}")

            for page_num in range(1, MAX_PAGES + 1):

                # Rotate user-agent per page
                ua = random.choice(USER_AGENTS)
                proxy = get_proxy()
                context_args = {
                    "user_agent": ua,
                    "viewport": {"width": 1280, "height": 800},
                }
                if proxy:
                    context_args["proxy"] = proxy

                context = await browser.new_context(**context_args)
                await context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                    window.chrome = { runtime: {} };
                """)
                page = await context.new_page()
                await Stealth().apply_stealth_async(page)

                url = (
                    f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
                    f"&page={page_num}"
                )
                print(f"\n  Page {page_num}/{MAX_PAGES} — UA: {ua[50:80]}...")

                success = await goto_with_retry(page, url)
                if not success:
                    failed_pages.append(url)
                    await context.close()
                    continue

                # Randomized delay — harder to detect
                await asyncio.sleep(random.uniform(5, 9))

                # Scroll to trigger lazy-loaded content
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                await asyncio.sleep(random.uniform(1.5, 3))
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(random.uniform(1.5, 3))

                product_links = await get_product_links(page)
                product_links = [
                    l for l in product_links
                    if '/dp/' in l and 'amazon.com' in l and l not in seen_urls
                ][:8]

                print(f"  Found {len(product_links)} new product links")

                if len(product_links) == 0:
                    await page.screenshot(
                        path=f"debug_{query.replace(' ','_')}_p{page_num}.png"
                    )
                    print(f"  ⚠ 0 links — screenshot saved. Check for CAPTCHA.")
                    await context.close()
                    continue

                for link in product_links:
                    seen_urls.add(link)
                    try:
                        print(f"  → {link[:70]}...")
                        success = await goto_with_retry(page, link)
                        if not success:
                            failed_pages.append(link)
                            continue

                        await asyncio.sleep(random.uniform(5, 10))

                        # Extract ASIN from URL for deduplication
                        asin = None
                        if '/dp/' in link:
                            asin = link.split('/dp/')[1].split('/')[0].split('?')[0]

                        title = await safe_get(
                            page, '#productTitle', 'el => el.innerText.trim()')
                        price = await safe_get(
                            page, '.a-price .a-offscreen', 'el => el.innerText.trim()')
                        if not price:
                            price = await safe_get(
                                page,
                                '#corePriceDisplay_desktop_feature_div .a-offscreen',
                                'el => el.innerText.trim()')
                        rating = await safe_get(
                            page, '#acrPopover', 'el => el.getAttribute("title")')
                        review_count = await safe_get(
                            page, '#acrCustomerReviewText', 'el => el.innerText.trim()')
                        bullets = await safe_get_all(
                            page, '#feature-bullets li span.a-list-item',
                            'els => els.map(e => e.innerText.trim())')
                        reviews = await safe_get_all(
                            page, 'span[data-hook="review-body"] span',
                            'els => els.map(e => e.innerText.trim())')

                        if not title:
                            print(f"     ✗ No title, skipping")
                            continue

                        product = {
                            "asin":         asin,
                            "title":        title,
                            "price":        price,
                            "rating":       rating,
                            "review_count": review_count,
                            "description":  " ".join(bullets) if bullets else "",
                            "reviews":      reviews[:5] if reviews else [],
                            "source_url":   link,
                            "category":     query,
                        }
                        products.append(product)
                        save_progress(products)
                        print(f"     ✓ [{len(products)}] {title[:50]}")

                    except Exception as e:
                        print(f"     ✗ Failed: {e}")
                        failed_pages.append(link)
                        continue

                await context.close()

        await browser.close()

    # ─── FINAL SUMMARY ────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  SCRAPE COMPLETE")
    print(f"  Total products : {len(products)}")
    print(f"  Categories     : {len(SEARCH_QUERIES)}")
    print(f"  Failed pages   : {len(failed_pages)}")
    print(f"  Output         : {OUTPUT_FILE}")
    print(f"{'='*58}")

    if failed_pages:
        print("\n  Failed URLs logged to data/raw/failed_pages.txt")
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        with open("data/raw/failed_pages.txt", "w") as f:
            f.write("\n".join(failed_pages))

if __name__ == "__main__":
    asyncio.run(scrape_amazon())