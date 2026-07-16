"""Dataset shape/size analysis for the recipe engine. Pure stdlib, never raises."""
from __future__ import annotations

import os

# ~45 art-style vocabulary terms. Lowercased; matched as whole-word/substring.
_STYLE_VOCAB = frozenset({
    "anime", "manga", "cartoon", "comic", "watercolor", "watercolour",
    "oil painting", "acrylic", "gouache", "pastel", "charcoal", "sketch",
    "line art", "lineart", "ink drawing", "pencil drawing", "3d render",
    "3d model", "octane render", "unreal engine", "pixel art", "voxel",
    "low poly", "vaporwave", "synthwave", "cyberpunk", "steampunk",
    "art nouveau", "art deco", "bauhaus", "impressionist", "impressionism",
    "surrealism", "surrealist", "cubism", "cubist", "baroque", "renaissance",
    "ukiyo-e", "woodblock", "pop art", "concept art", "matte painting",
    "digital painting", "flat design", "isometric", "cel shaded", "cel-shaded",
    "chibi", "graffiti", "psychedelic", "minimalist", "photorealistic style",
    "storybook", "fantasy art", "sci-fi art", "vector art", "risograph",
})


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())


def count_images(dir_: str) -> int:
    """Count image files (png/jpg/jpeg/webp) directly under dir_. On error -> 0."""
    try:
        exts = (".png", ".jpg", ".jpeg", ".webp")
        return sum(1 for e in os.listdir(dir_) if e.lower().endswith(exts))
    except Exception:
        return 0


def detect_shape(captions: list[str]) -> str:
    """'style' if a style-vocab term appears in >=25% of non-empty captions,
    else 'subject'. Defensive default on any error or empty input: 'style'
    (the lower-risk optimizer path — AdamW never diverges; a misrouted subject
    still trains, whereas Prodigy on a true style is the riskier miss)."""
    try:
        caps = [_norm(c) for c in captions if c and c.strip()]
        if not caps:
            return "style"
        hits = sum(1 for c in caps if any(term in c for term in _STYLE_VOCAB))
        return "style" if (hits / len(caps)) >= 0.25 else "subject"
    except Exception:
        return "style"


def size_bucket(n: int) -> str:
    """ECH<=10, CH<=20, M<=30, G<=50, EG>50. On error -> 'M' (mid, safe epochs)."""
    try:
        n = int(n)
        if n <= 10:
            return "ECH"
        if n <= 20:
            return "CH"
        if n <= 30:
            return "M"
        if n <= 50:
            return "G"
        return "EG"
    except Exception:
        return "M"
