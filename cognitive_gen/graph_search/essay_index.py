"""
Essay index with metadata and similarity-based neighbor finding.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from .scraper import BlogScraper, EssayData


@dataclass
class EssayMetadata:
    """Metadata for an essay."""
    slug: str
    title: str
    word_count: int
    keywords: list[str] = field(default_factory=list)
    avg_sentence_length: float = 0.0


class EssayIndex:
    """
    Index of essays with similarity computation for neighbor finding.

    Uses keyword overlap for similarity (simple but effective).
    """

    def __init__(self, scraper: Optional[BlogScraper] = None):
        self.scraper = scraper or BlogScraper()
        self.essays: dict[str, EssayData] = {}
        self.metadata: dict[str, EssayMetadata] = {}
        self._similarity_cache: dict[tuple[str, str], float] = {}

    def load(self):
        """Load all essays and compute metadata."""
        self.essays = self.scraper.load_all_essays()

        for slug, essay in self.essays.items():
            self.metadata[slug] = self._compute_metadata(essay)

        print(f"Indexed {len(self.essays)} essays")

    def _compute_metadata(self, essay: EssayData) -> EssayMetadata:
        """Compute metadata for an essay."""
        # Extract keywords (simple: most common words, excluding stopwords)
        words = re.findall(r'\b[a-z]{4,}\b', essay.content.lower())

        stopwords = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were',
            'which', 'their', 'would', 'could', 'should', 'there', 'where',
            'what', 'when', 'about', 'into', 'more', 'some', 'than', 'them',
            'then', 'these', 'those', 'will', 'your', 'also', 'being', 'does',
            'each', 'even', 'just', 'like', 'make', 'only', 'over', 'such',
            'very', 'after', 'before', 'between', 'both', 'come', 'first',
            'good', 'great', 'know', 'most', 'other', 'people', 'same',
            'think', 'through', 'want', 'well', 'work', 'because', 'something',
        }

        filtered = [w for w in words if w not in stopwords]
        word_counts = Counter(filtered)
        keywords = [word for word, _ in word_counts.most_common(20)]

        # Compute average sentence length
        sentences = re.split(r'[.!?]+', essay.content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            avg_sentence_length = 0.0

        return EssayMetadata(
            slug=essay.slug,
            title=essay.title,
            word_count=essay.word_count,
            keywords=keywords,
            avg_sentence_length=avg_sentence_length,
        )

    def get_essay(self, slug: str) -> Optional[str]:
        """Get essay content by slug."""
        if slug in self.essays:
            return self.essays[slug].content
        return None

    def get_essay_data(self, slug: str) -> Optional[EssayData]:
        """Get full essay data by slug."""
        return self.essays.get(slug)

    def compute_similarity(self, slug_a: str, slug_b: str) -> float:
        """
        Compute similarity between two essays using keyword overlap.

        Returns a value between 0 (no similarity) and 1 (identical keywords).
        """
        if slug_a == slug_b:
            return 1.0

        # Check cache
        cache_key = tuple(sorted([slug_a, slug_b]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        meta_a = self.metadata.get(slug_a)
        meta_b = self.metadata.get(slug_b)

        if not meta_a or not meta_b:
            return 0.0

        # Jaccard similarity on keywords
        set_a = set(meta_a.keywords)
        set_b = set(meta_b.keywords)

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        similarity = intersection / union if union > 0 else 0.0

        # Cache result
        self._similarity_cache[cache_key] = similarity

        return similarity

    def get_related_essays(self, slug: str, n: int = 3) -> list[str]:
        """
        Find n most similar essays to the given one.

        Args:
            slug: Essay to find neighbors for
            n: Number of neighbors to return

        Returns:
            List of essay slugs, most similar first
        """
        if slug not in self.essays:
            return []

        # Compute similarity to all other essays
        similarities = []
        for other_slug in self.essays:
            if other_slug != slug:
                sim = self.compute_similarity(slug, other_slug)
                similarities.append((other_slug, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in similarities[:n]]

    def get_all_slugs(self) -> list[str]:
        """Get list of all essay slugs."""
        return list(self.essays.keys())

    def split_essay(self, slug: str, context_ratio: float = 0.6) -> tuple[str, str]:
        """
        Split essay into context and target regions.

        Args:
            slug: Essay slug
            context_ratio: Fraction of essay to use as context (default 60%)

        Returns:
            (context_region, target_region)
        """
        content = self.get_essay(slug)
        if not content:
            return "", ""

        # Split by character position
        split_point = int(len(content) * context_ratio)

        # Try to split at a paragraph or sentence boundary
        # Look for paragraph break near split point
        search_start = max(0, split_point - 200)
        search_end = min(len(content), split_point + 200)
        search_region = content[search_start:search_end]

        # Find paragraph breaks
        para_breaks = [i for i, c in enumerate(search_region) if c == '\n' and i > 0 and search_region[i-1] == '\n']

        if para_breaks:
            # Use closest paragraph break to desired split point
            desired_pos = split_point - search_start
            best_break = min(para_breaks, key=lambda x: abs(x - desired_pos))
            actual_split = search_start + best_break
        else:
            # Fall back to sentence boundary
            sentence_ends = [i for i, c in enumerate(search_region) if c in '.!?' and i < len(search_region) - 1]
            if sentence_ends:
                desired_pos = split_point - search_start
                best_break = min(sentence_ends, key=lambda x: abs(x - desired_pos))
                actual_split = search_start + best_break + 1
            else:
                actual_split = split_point

        context = content[:actual_split].strip()
        target = content[actual_split:].strip()

        return context, target

    def get_target_chunks(self, slug: str, n_chunks: int = 3, context_ratio: float = 0.6) -> list[str]:
        """
        Split the target region of an essay into chunks for rotating targets.

        Args:
            slug: Essay slug
            n_chunks: Number of chunks to create
            context_ratio: Fraction used for context (target is the rest)

        Returns:
            List of target chunks
        """
        _, target_region = self.split_essay(slug, context_ratio)

        if not target_region:
            return []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in target_region.split('\n\n') if p.strip()]

        if len(paragraphs) <= n_chunks:
            return paragraphs if paragraphs else [target_region]

        # Group paragraphs into n_chunks
        chunk_size = len(paragraphs) // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            if i == n_chunks - 1:
                # Last chunk gets remainder
                chunk_paragraphs = paragraphs[start:]
            else:
                chunk_paragraphs = paragraphs[start:start + chunk_size]
            chunks.append('\n\n'.join(chunk_paragraphs))

        return chunks


def create_index(cache_dir: Optional[Path] = None) -> EssayIndex:
    """
    Convenience function to create and load an essay index.

    Args:
        cache_dir: Optional custom cache directory

    Returns:
        Loaded EssayIndex
    """
    scraper = BlogScraper(cache_dir)
    index = EssayIndex(scraper)
    index.load()
    return index


if __name__ == "__main__":
    # Test index
    index = create_index()

    print("\nEssay similarities:")
    for slug in list(index.essays.keys())[:5]:
        neighbors = index.get_related_essays(slug, n=3)
        print(f"\n{slug}:")
        for neighbor in neighbors:
            sim = index.compute_similarity(slug, neighbor)
            print(f"  -> {neighbor}: {sim:.3f}")

    print("\n\nSample split (shame):")
    context, target = index.split_essay('shame')
    print(f"Context ({len(context)} chars): {context[:200]}...")
    print(f"Target ({len(target)} chars): {target[:200]}...")

    chunks = index.get_target_chunks('shame', n_chunks=3)
    print(f"\nTarget chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars")
