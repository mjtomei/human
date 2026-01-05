"""
Blog scraper with caching for mtomei.com essays.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING = True
except ImportError:
    HAS_SCRAPING = False


@dataclass
class EssayData:
    """Data for a single essay."""
    slug: str
    title: str
    content: str
    url: str
    word_count: int
    scraped_at: Optional[str] = None

    @classmethod
    def from_file(cls, path: Path) -> 'EssayData':
        """Load essay from a cached text file."""
        slug = path.stem
        content = path.read_text(encoding='utf-8').strip()
        word_count = len(content.split())

        return cls(
            slug=slug,
            title=slug.replace('_', ' ').title(),
            content=content,
            url=f"https://mtomei.com/{slug}",
            word_count=word_count,
            scraped_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class BlogScraper:
    """
    Scrape essays from mtomei.com with local caching.

    Uses cached files when available, falls back to web scraping if needed.
    """

    DEFAULT_CACHE_DIR = Path("/tmp/mtomei_essays")
    BASE_URL = "https://mtomei.com"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if HAS_SCRAPING:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (cognitive research project)'
            })
        else:
            self.session = None

    def get_cached_slugs(self) -> list[str]:
        """Get list of essay slugs available in cache."""
        slugs = []
        for path in self.cache_dir.glob("*.txt"):
            slugs.append(path.stem)
        return sorted(slugs)

    def load_from_cache(self, slug: str) -> Optional[EssayData]:
        """Load an essay from cache if it exists."""
        cache_path = self.cache_dir / f"{slug}.txt"
        if cache_path.exists():
            return EssayData.from_file(cache_path)
        return None

    def save_to_cache(self, essay: EssayData) -> Path:
        """Save an essay to cache."""
        cache_path = self.cache_dir / f"{essay.slug}.txt"
        cache_path.write_text(essay.content, encoding='utf-8')
        return cache_path

    def scrape_essay(self, slug: str) -> Optional[EssayData]:
        """
        Scrape a single essay from the web.

        Returns None if scraping is not available or fails.
        """
        if not HAS_SCRAPING:
            print(f"Warning: requests/beautifulsoup not available, cannot scrape {slug}")
            return None

        url = f"{self.BASE_URL}/{slug}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to extract main content
            # This may need adjustment based on actual site structure
            content_elem = (
                soup.find('article') or
                soup.find('main') or
                soup.find('div', class_='content') or
                soup.find('div', class_='post')
            )

            if content_elem:
                # Remove script and style elements
                for elem in content_elem.find_all(['script', 'style', 'nav']):
                    elem.decompose()
                content = content_elem.get_text(separator='\n\n', strip=True)
            else:
                # Fallback: get body text
                content = soup.body.get_text(separator='\n\n', strip=True) if soup.body else ""

            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else slug.title()

            essay = EssayData(
                slug=slug,
                title=title,
                content=content,
                url=url,
                word_count=len(content.split()),
                scraped_at=datetime.now().isoformat(),
            )

            return essay

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def get_essay(self, slug: str, force_scrape: bool = False) -> Optional[EssayData]:
        """
        Get an essay, using cache if available.

        Args:
            slug: Essay identifier
            force_scrape: If True, scrape even if cached

        Returns:
            EssayData or None if not found
        """
        if not force_scrape:
            cached = self.load_from_cache(slug)
            if cached:
                return cached

        # Try to scrape
        essay = self.scrape_essay(slug)
        if essay:
            self.save_to_cache(essay)
            return essay

        return None

    def load_all_essays(self, force_scrape: bool = False) -> dict[str, EssayData]:
        """
        Load all available essays.

        First loads from cache, then optionally scrapes missing essays.

        Returns:
            Dictionary mapping slug to EssayData
        """
        essays = {}

        # Load all cached essays
        for slug in self.get_cached_slugs():
            essay = self.load_from_cache(slug)
            if essay:
                essays[slug] = essay

        print(f"Loaded {len(essays)} essays from cache")
        return essays

    def get_essay_list_from_web(self) -> list[str]:
        """
        Try to get list of all essay slugs from the website.

        This would scrape the index/sitemap to find all essays.
        Returns empty list if scraping not available.
        """
        if not HAS_SCRAPING:
            return []

        # This would need to be implemented based on actual site structure
        # For now, return empty - we'll rely on cached essays
        return []


def load_essays(cache_dir: Optional[Path] = None) -> dict[str, EssayData]:
    """
    Convenience function to load all cached essays.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        Dictionary mapping slug to EssayData
    """
    scraper = BlogScraper(cache_dir)
    return scraper.load_all_essays()


if __name__ == "__main__":
    # Test loading
    scraper = BlogScraper()
    essays = scraper.load_all_essays()

    print(f"\nLoaded {len(essays)} essays:")
    for slug, essay in sorted(essays.items()):
        print(f"  {slug}: {essay.word_count} words")

    # Show total
    total_words = sum(e.word_count for e in essays.values())
    print(f"\nTotal: {total_words} words")
