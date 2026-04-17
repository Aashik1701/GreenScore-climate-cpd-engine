import os
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def wait_for_ready(page):
    page.wait_for_timeout(4000)
    try:
        page.get_by_text("Loaded", exact=False).first.wait_for(timeout=45000)
        page.wait_for_timeout(1500)
    except PlaywrightTimeoutError:
        page.wait_for_timeout(3000)


def safe_click_text(page, text, timeout=10000):
    locator = page.get_by_text(text, exact=False).first
    locator.wait_for(timeout=timeout)
    locator.click()


def capture(url: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(url, wait_until="domcontentloaded", timeout=120000)

        page.wait_for_timeout(2000)
        page.screenshot(path=str(out / "01_landing.png"), full_page=True)

        safe_click_text(page, "Select Dataset")
        safe_click_text(page, "LendingClub (US)")
        wait_for_ready(page)
        page.screenshot(path=str(out / "02_overview.png"), full_page=True)

        safe_click_text(page, "Multi-Scenario")
        page.wait_for_timeout(5000)
        page.screenshot(path=str(out / "03_multi_scenario.png"), full_page=True)

        safe_click_text(page, "Heatmap")
        page.wait_for_timeout(4000)
        page.screenshot(path=str(out / "04_heatmap.png"), full_page=True)

        safe_click_text(page, "Expected Loss")
        page.wait_for_timeout(4000)
        page.screenshot(path=str(out / "05_expected_loss.png"), full_page=True)

        browser.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture GreenScore dashboard screenshots")
    parser.add_argument("--url", default="http://localhost:8501", help="Dashboard URL")
    parser.add_argument("--out", default="docs/media", help="Output directory")
    args = parser.parse_args()

    capture(args.url, args.out)

    # Optional GIF assembly
    try:
        from PIL import Image

        frame_files = [
            "01_landing.png",
            "02_overview.png",
            "03_multi_scenario.png",
            "04_heatmap.png",
            "05_expected_loss.png",
        ]
        frames = []
        for name in frame_files:
            p = os.path.join(args.out, name)
            if os.path.exists(p):
                frames.append(Image.open(p).convert("RGB"))

        if len(frames) >= 2:
            gif_path = os.path.join(args.out, "dashboard_tour.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1500,
                loop=0,
            )
            print(f"Created GIF: {gif_path}")
    except Exception as e:
        print(f"GIF creation skipped: {e}")
