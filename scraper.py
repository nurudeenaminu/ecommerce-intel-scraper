import asyncio
import json
import random
from pathlib import Path
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

SEARCH_QUERY = "wireless headphones"
MAX_PAGES = 3
OUTPUT_FILE = "data/raw/products_raw.json"

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
    """Try multiple strategies to extract /dp/ product links."""
    # Strategy 1: data-asin attribute (most reliable across Amazon updates)
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

    # Strategy 2: any anchor with /dp/ in href (broad fallback)
    links = await page.evaluate("""
        () => {
            const all = document.querySelectorAll('a[href*="/dp/"]');
            const hrefs = Array.from(all)
                .map(a => a.href)
                .filter(h => h.includes('amazon.com') && h.includes('/dp/'));
            return [...new Set(hrefs)];
        }
    """)
    return links

async def scrape_amazon():
    products = []

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
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            window.chrome = { runtime: {} };
        """)

        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        for page_num in range(1, MAX_PAGES + 1):
            url = (
                f"https://www.amazon.com/s?k={SEARCH_QUERY.replace(' ', '+')}"
                f"&page={page_num}"
            )
            print(f"\nScraping search page {page_num}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Give JS time to fully render the results
            await asyncio.sleep(random.uniform(5, 8))

            # Scroll down to trigger lazy-loaded content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(2)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

            product_links = await get_product_links(page)

            # Filter out non-product links (sponsored, ads, accessories)
            product_links = [
                l for l in product_links
                if '/dp/' in l and 'amazon.com' in l
            ][:8]

            print(f"  Found {len(product_links)} product links")

            if len(product_links) == 0:
                # Save screenshot for debugging
                await page.screenshot(path=f"debug_page_{page_num}.png")
                print(f"  ⚠ 0 links found. Screenshot saved: debug_page_{page_num}.png")
                print(f"  Page title: {await page.title()}")
                continue

            for link in product_links:
                try:
                    print(f"  → {link[:70]}...")
                    await page.goto(link, wait_until="domcontentloaded", timeout=30000)
                    await asyncio.sleep(random.uniform(5, 9))

                    title = await safe_get(
                        page, '#productTitle', 'el => el.innerText.trim()')
                    price = await safe_get(
                        page, '.a-price .a-offscreen', 'el => el.innerText.trim()')
                    if not price:
                        price = await safe_get(
                            page, '#corePriceDisplay_desktop_feature_div .a-offscreen',
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
                        print(f"     ✗ No title found, skipping")
                        continue

                    product = {
                        "title":        title,
                        "price":        price,
                        "rating":       rating,
                        "review_count": review_count,
                        "description":  " ".join(bullets) if bullets else "",
                        "reviews":      reviews[:5] if reviews else [],
                        "source_url":   link,
                        "category":     SEARCH_QUERY,
                    }
                    products.append(product)
                    print(f"     ✓ {title[:55]}")

                except Exception as e:
                    print(f"     ✗ Failed: {e}")
                    continue

        await browser.close()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Done. {len(products)} products saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(scrape_amazon())