import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

async def diagnose():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=50)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        url = "https://www.amazon.com/s?k=wireless+headphones&page=1"
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Wait longer for JS to render
        await asyncio.sleep(5)

        # Save screenshot so you can see exactly what loaded
        await page.screenshot(path="debug_search.png", full_page=True)
        print("Screenshot saved: debug_search.png")

        # Try every possible selector Amazon uses
        selectors_to_try = [
            '[data-component-type="s-search-result"] h2 a',
            '.s-result-item h2 a',
            '.s-search-results h2 a',
            'h2.a-size-mini a',
            'h2 a.a-link-normal',
            '[data-asin] h2 a',
            '.sg-col-inner h2 a',
        ]

        for sel in selectors_to_try:
            try:
                els = await page.query_selector_all(sel)
                hrefs = []
                for el in els:
                    href = await el.get_attribute('href')
                    if href and '/dp/' in href:
                        hrefs.append(href)
                print(f"Selector '{sel}': {len(els)} elements, {len(hrefs)} with /dp/")
            except Exception as e:
                print(f"Selector '{sel}': ERROR - {e}")

        # Also dump ALL /dp/ links on the page regardless of selector
        all_dp_links = await page.evaluate("""
            () => {
                const all = document.querySelectorAll('a[href*="/dp/"]');
                return Array.from(all).map(a => a.href).filter(h => h.includes('/dp/'));
            }
        """)
        unique_dp = list(set(all_dp_links))
        print(f"\nTotal /dp/ links found anywhere on page: {len(unique_dp)}")
        for link in unique_dp[:10]:
            print(f"  {link[:90]}")

        # Print page title to confirm what loaded
        print(f"\nPage title: {await page.title()}")

        await browser.close()

asyncio.run(diagnose())