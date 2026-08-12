#!/usr/bin/env python3
"""Build rights-clean, deterministic Week-7 HKE fixture candidates.

This module is calibration-only.  It uses only integer geometry, colors, and a
small code-owned bitmap alphabet.  It performs no network access, reads no
ambient configuration, uses no external asset or font, and never represents an
agent-generated record as human approval.

Discovery bytes and their manifest are written to one create-only output.  The
confirmation bytes and their full manifest are written to a distinct,
explicitly supplied custodian output.  The public manifest contains only each
confirmation set's cardinality and semantic commitment; it contains no
confirmation row identity, path, caption, parameter, or byte hash.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from io import BytesIO
import json
import os
import platform
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
import unicodedata

import PIL
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True

SCHEMA = 3
KIND = "sn56-week7-hke-procedural-candidate"
RENDERER_VERSION = "3.0.0"
GENERATOR_REPOSITORY = "https://github.com/tuly1/sn56-forge-toolkit.git"
GENERATOR_SOURCE_PATH = "ops/experiments/week7/hke_procedural_renderer.py"
DECLARATIVE_CONTRACT_PATH = Path(__file__).with_name("hke_fixture_contract.json")
DECLARATIVE_CONTRACT_SOURCE_PATH = "ops/experiments/week7/hke_fixture_contract.json"
DISCOVERY_DOMAIN = b"SN56-W7-HKE-DISCOVERY-v3"
CONFIRMATION_DOMAIN = b"SN56-W7-HKE-CONFIRMATION-v3"
PNG_FORMAT = "PNG"
SCRIPT_PATH = Path(__file__).resolve()
EXECUTABLE_REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
FIXED_GIT = Path("/usr/bin/git")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_DIRECTORY", 0)
)

FIXTURE_CONTRACT: tuple[dict[str, Any], ...] = (
    {
        "fixture_id": "W7-HKE-SOCIAL-A",
        "family": "social",
        "discovery_packs": [
            {"pack": "D1", "training_count": 10, "evaluation_count": 8},
            {"pack": "D2", "training_count": 10, "evaluation_count": 8},
        ],
        "confirmation_packs": [
            {"pack": "C1", "training_count": 10, "evaluation_count": 8},
            {"pack": "C2", "training_count": 10, "evaluation_count": 8},
        ],
        "discovery_count": 36,
        "confirmation_count": 36,
    },
    {
        "fixture_id": "W7-HKE-PRODUCT-A",
        "family": "product",
        "discovery_packs": [
            {"pack": "D1", "training_count": 10, "evaluation_count": 8}
        ],
        "confirmation_packs": [
            {"pack": "C1", "training_count": 10, "evaluation_count": 8}
        ],
        "discovery_count": 18,
        "confirmation_count": 18,
    },
    {
        "fixture_id": "W7-HKE-LOGO-UI-A",
        "family": "logo_ui",
        "discovery_packs": [
            {"pack": "D1", "training_count": 10, "evaluation_count": 8}
        ],
        "confirmation_packs": [
            {"pack": "C1", "training_count": 10, "evaluation_count": 8}
        ],
        "discovery_count": 18,
        "confirmation_count": 18,
    },
)

FORBIDDEN_INVENTORIES = {
    "external_assets": [],
    "external_fonts": [],
    "provider_outputs": [],
    "reference_images": [],
    "tournament_or_opponent_content": [],
}

RIGHTS_DECLARATION = {
    "classification": "first-party-procedural-candidate",
    "generator_assets": "integer-primitives-and-code-owned-bitmap-glyphs-only",
    "external_assets_used": False,
    "external_fonts_used": False,
    "provider_outputs_used": False,
    "reference_images_used": False,
    "tournament_or_opponent_content_used": False,
    "human_rights_review": "not_performed",
    "training_and_evaluation_use_decision": "pending_human_review",
}

RIGHTS_RECORD_KIND = "sn56-week7-hke-first-party-rights-record"

_ROW_KEYS = {
    "row_id",
    "fixture_id",
    "family",
    "phase",
    "pack",
    "split_role",
    "ordinal",
    "relative_image_path",
    "relative_caption_path",
    "parameters",
    "parameters_sha256",
    "seed_commitment_sha256",
    "image_sha256",
    "image_bytes",
    "decoded_pixels_sha256",
    "caption_sha256",
    "caption_bytes",
    "normalized_caption_sha256",
    "width",
    "height",
    "format",
    "mode",
    "visible_glyph_transcript",
    "group_identity",
    "group_identity_sha256",
    "rights_declaration",
    "row_record_sha256",
}

_PALETTES: tuple[tuple[tuple[int, int, int], ...], ...] = (
    ((19, 30, 53), (242, 245, 250), (47, 111, 237), (242, 92, 84), (132, 224, 198)),
    ((36, 27, 58), (250, 246, 240), (119, 79, 240), (241, 151, 58), (93, 198, 182)),
    ((16, 48, 45), (243, 248, 239), (24, 145, 132), (230, 99, 76), (240, 196, 72)),
    ((45, 36, 28), (249, 244, 235), (59, 114, 176), (199, 82, 98), (113, 176, 123)),
)

# Original 5x7 bitmap glyphs.  No system or bundled font is loaded.
_GLYPHS: dict[str, tuple[str, ...]] = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
}


class FixtureError(RuntimeError):
    """Fail-closed procedural fixture error."""


def _git_sha(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 40:
        raise FixtureError(f"{label} must be an exact 40-hex identity")
    try:
        int(text, 16)
    except ValueError as exc:
        raise FixtureError(f"{label} must be an exact 40-hex identity") from exc
    return text


def _identity_text(value: Any, label: str) -> str:
    text = " ".join(str(value or "").split())
    if (
        len(text) < 3
        or len(text) > 256
        or text.casefold()
        in {
            "unknown",
            "pending",
            "placeholder",
            "tbd",
        }
    ):
        raise FixtureError(f"{label} must be an explicit 3..256 character record")
    return text


def _rights_record(
    *, author_record: Any, rights_owner: Any, license_or_use_grant: Any
) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "kind": RIGHTS_RECORD_KIND,
        "author_record": _identity_text(author_record, "author record"),
        "rights_owner": _identity_text(rights_owner, "rights owner"),
        "license_or_use_grant": _identity_text(
            license_or_use_grant, "license or use grant"
        ),
        "generation_basis": dict(RIGHTS_DECLARATION),
        "named_human_review": "not_performed",
    }
    return {**body, "semantic_sha256": semantic_sha256(body)}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(_read_regular(SCRIPT_PATH, "renderer source")).hexdigest()


def _contract_source_sha256() -> str:
    return hashlib.sha256(
        _read_regular(DECLARATIVE_CONTRACT_PATH, "declarative fixture contract")
    ).hexdigest()


def _validate_key(value: bytes | bytearray, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)) or not 32 <= len(value) <= 4096:
        raise FixtureError(f"{label} must contain 32..4096 bytes")
    return bytes(value)


def _require_distinct_phase_keys(discovery_key: bytes, confirmation_key: bytes) -> None:
    if hmac.compare_digest(discovery_key, confirmation_key):
        raise FixtureError("discovery and confirmation keys must be distinct")


def _phase_packs(fixture: Mapping[str, Any], phase: str) -> Sequence[Mapping[str, Any]]:
    packs = fixture.get(f"{phase}_packs")
    if not isinstance(packs, (tuple, list)) or not packs:
        raise FixtureError(f"{fixture.get('fixture_id')} {phase} packs are invalid")
    return packs


def _pack_assignment(
    fixture: Mapping[str, Any], phase: str, ordinal: int
) -> tuple[str, str, int]:
    if ordinal < 0:
        raise FixtureError("row ordinal cannot be negative")
    offset = ordinal
    for record in _phase_packs(fixture, phase):
        pack = str(record.get("pack", ""))
        training_count = record.get("training_count")
        evaluation_count = record.get("evaluation_count")
        if (
            not pack
            or isinstance(training_count, bool)
            or not isinstance(training_count, int)
            or training_count <= 0
            or isinstance(evaluation_count, bool)
            or not isinstance(evaluation_count, int)
            or evaluation_count <= 0
        ):
            raise FixtureError(f"{fixture.get('fixture_id')} {phase} pack is invalid")
        if offset < training_count:
            return pack, "training", offset
        offset -= training_count
        if offset < evaluation_count:
            return pack, "evaluation", offset
        offset -= evaluation_count
    raise FixtureError(f"{fixture.get('fixture_id')} {phase} ordinal is out of range")


def _derive(
    key: bytes,
    domain: bytes,
    fixture_id: str,
    pack: str,
    split_role: str,
    split_ordinal: int,
) -> bytes:
    message = (
        domain
        + b"\0"
        + fixture_id.encode("ascii")
        + b"\0"
        + pack.encode("ascii")
        + b"\0"
        + split_role.encode("ascii")
        + b"\0"
        + split_ordinal.to_bytes(4, "big")
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def _number(seed: bytes, label: str, modulus: int) -> int:
    if modulus <= 0:
        raise AssertionError("modulus must be positive")
    digest = hmac.new(seed, label.encode("ascii"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _phase_key_commitment(key: bytes, domain: bytes) -> str:
    return hmac.new(key, domain + b"\0key-commitment", hashlib.sha256).hexdigest()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    scale: int,
) -> None:
    x, y = position
    for character in text.upper():
        glyph = _GLYPHS.get(character)
        if glyph is None:
            raise FixtureError(f"unsupported code-owned glyph: {character!r}")
        for row, pattern in enumerate(glyph):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    left = x + column * scale
                    top = y + row * scale
                    draw.rectangle(
                        (left, top, left + scale - 1, top + scale - 1), fill=color
                    )
        x += 6 * scale


def _row_parameters(
    fixture: Mapping[str, Any], phase: str, ordinal: int, key: bytes
) -> dict[str, Any]:
    domain = DISCOVERY_DOMAIN if phase == "discovery" else CONFIRMATION_DOMAIN
    pack, split_role, split_ordinal = _pack_assignment(fixture, phase, ordinal)
    seed = _derive(
        key,
        domain,
        str(fixture["fixture_id"]),
        pack,
        split_role,
        split_ordinal,
    )
    family = fixture["family"]
    subtype = None
    if family == "logo_ui":
        subtype = "logo" if split_ordinal % 2 == 0 else "ui"
    dimensions = {
        "social": ((1024, 1024), (1024, 1280), (1280, 720)),
        "product": ((1024, 1024), (1152, 864)),
        "logo_ui": ((1024, 1024), (1280, 800)),
    }[family]
    return {
        "palette": _number(seed, "palette", len(_PALETTES)),
        # Three independently derived channel offsets vary the dominant color
        # field across rows.  This is not decorative entropy: without it,
        # legitimate members of one product/UI grammar collapse under the
        # admission screen's dHash+MSE near-duplicate rule.
        "color_jitter": [
            _number(seed, "color-jitter-r", 81) - 40,
            _number(seed, "color-jitter-g", 81) - 40,
            _number(seed, "color-jitter-b", 81) - 40,
        ],
        "layout": _number(seed, "layout", 7),
        "accent_shift": _number(seed, "accent-shift", 97),
        "shape_a": _number(seed, "shape-a", 211),
        "shape_b": _number(seed, "shape-b", 223),
        "field_pattern": [
            _number(seed, f"field-pattern-{index:02d}", len(_PALETTES[0]))
            for index in range(32)
        ],
        "dimensions": list(dimensions[split_ordinal % len(dimensions)]),
        "subtype": subtype,
        "pack": pack,
        "split_role": split_role,
        "split_ordinal": split_ordinal,
        "phase_domain": phase,
    }


def _row_palette(params: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    base = _PALETTES[int(params["palette"])]
    jitter = [int(value) for value in params["color_jitter"]]
    if len(jitter) != 3:
        raise FixtureError("row color jitter must have three channels")
    return tuple(
        tuple(
            max(0, min(255, channel + jitter[index]))
            for index, channel in enumerate(color)
        )
        for color in base
    )


def _draw_unique_field(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    params: Mapping[str, Any],
    palette: tuple[tuple[int, int, int], ...],
) -> None:
    """Draw a row-unique 8x4 field before the semantic foreground.

    Shared product/UI grammars otherwise create automated near-duplicate
    clusters.  We preserve the screen and make the intended row variation
    visible instead of weakening its thresholds.
    """

    width, height = size
    pattern = [int(value) for value in params["field_pattern"]]
    if len(pattern) != 32 or any(not 0 <= value < len(palette) for value in pattern):
        raise FixtureError("row field pattern is invalid")
    for row in range(4):
        for column in range(8):
            left = column * width // 8
            right = (column + 1) * width // 8
            top = row * height // 4
            bottom = (row + 1) * height // 4
            draw.rectangle(
                (left, top, right, bottom),
                fill=palette[pattern[row * 8 + column]],
            )


def _caption(family: str, display_token: str, subtype: str | None) -> str:
    if family == "social":
        return (
            "S7QAL SOCIAL, original procedural information card variation "
            f"{display_token}."
        )
    if family == "product":
        return (
            "P7VEX PRODUCT, original parametric desk luminaire variation "
            f"{display_token}."
        )
    return (
        f"L7MIR SYSTEM, original procedural {subtype} composition variation "
        f"{display_token}."
    )


def _draw_social(
    image: Image.Image, params: Mapping[str, Any], display_token: str
) -> list[str]:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    palette = _row_palette(params)
    draw.rectangle((0, 0, width - 1, height - 1), fill=palette[1])
    _draw_unique_field(draw, image.size, params, palette)
    margin = width // 12
    draw.rounded_rectangle(
        (margin, margin, width - margin, height - margin),
        radius=max(12, width // 32),
        fill=palette[0],
    )
    band_y = margin + height // 7
    draw.rectangle(
        (margin, band_y, width - margin, band_y + height // 5), fill=palette[2]
    )
    for index in range(5):
        x = margin + width // 12 + index * width // 7
        y = band_y + height // 3 + ((index + int(params["layout"])) % 3) * height // 28
        radius = width // 28 + (int(params["shape_a"]) + index * 7) % (width // 45)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=palette[3 + index % 2],
        )
    strings = ["SIGNAL", f"CARD {display_token}"]
    _draw_text(
        draw,
        (margin + width // 16, margin + height // 22),
        strings[0],
        palette[1],
        scale=max(5, width // 150),
    )
    _draw_text(
        draw,
        (margin + width // 16, band_y + height // 18),
        strings[1],
        palette[0],
        scale=max(4, width // 180),
    )
    return strings


def _draw_product(
    image: Image.Image, params: Mapping[str, Any], display_token: str
) -> list[str]:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    palette = _row_palette(params)
    draw.rectangle((0, 0, width - 1, height - 1), fill=palette[1])
    _draw_unique_field(draw, image.size, params, palette)
    horizon = height * 3 // 4
    draw.rectangle((0, horizon, width, height), fill=palette[4])
    cx = width // 2 + (int(params["shape_a"]) - 105) * width // 1800
    base_w = width // 4
    draw.ellipse(
        (cx - base_w, horizon - height // 24, cx + base_w, horizon + height // 20),
        fill=palette[0],
    )
    stem_w = max(12, width // 35)
    stem_top = height // 3 + (int(params["layout"]) - 3) * height // 90
    draw.rounded_rectangle(
        (cx - stem_w, stem_top, cx + stem_w, horizon), radius=stem_w, fill=palette[2]
    )
    shade_w = width // 5 + (int(params["shape_b"]) % (width // 15))
    shade_h = height // 7
    draw.polygon(
        (
            (cx - shade_w, stem_top + shade_h),
            (cx - shade_w // 2, stem_top),
            (cx + shade_w // 2, stem_top),
            (cx + shade_w, stem_top + shade_h),
        ),
        fill=palette[3],
    )
    draw.ellipse(
        (
            cx - shade_w // 2,
            stem_top + shade_h // 2,
            cx + shade_w // 2,
            stem_top + shade_h * 2,
        ),
        fill=palette[4],
    )
    strings = ["TAVORA", f"FORM {display_token}"]
    _draw_text(
        draw,
        (width // 16, height // 14),
        strings[0],
        palette[0],
        scale=max(5, width // 160),
    )
    _draw_text(
        draw,
        (width // 16, height // 14 + 11 * max(5, width // 160)),
        strings[1],
        palette[2],
        scale=max(4, width // 190),
    )
    return strings


def _draw_logo_ui(
    image: Image.Image, params: Mapping[str, Any], display_token: str
) -> list[str]:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    palette = _row_palette(params)
    subtype = str(params["subtype"])
    draw.rectangle((0, 0, width - 1, height - 1), fill=palette[1])
    _draw_unique_field(draw, image.size, params, palette)
    if subtype == "logo":
        cx, cy = width // 2, height * 2 // 5
        radius = width // 5
        shift = int(params["accent_shift"]) * width // 1200
        draw.polygon(
            (
                (cx, cy - radius),
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
            ),
            fill=palette[2],
        )
        draw.ellipse(
            (
                cx - radius // 2 + shift,
                cy - radius // 2,
                cx + radius // 2 + shift,
                cy + radius // 2,
            ),
            fill=palette[3],
        )
        strings = ["LUMERA", f"MARK {display_token}"]
        _draw_text(
            draw,
            (width // 2 - width // 5, height * 3 // 4),
            strings[0],
            palette[0],
            scale=max(5, width // 150),
        )
        _draw_text(
            draw,
            (width // 2 - width // 6, height * 3 // 4 + height // 12),
            strings[1],
            palette[2],
            scale=max(4, width // 190),
        )
    else:
        sidebar = width // 5
        draw.rectangle((0, 0, sidebar, height), fill=palette[0])
        draw.rectangle(
            (sidebar + width // 20, height // 8, width - width // 20, height // 3),
            fill=palette[2],
        )
        for index in range(3):
            left = sidebar + width // 20 + index * width // 4
            top = height * 2 // 5
            draw.rounded_rectangle(
                (left, top, left + width // 5, top + height // 3),
                radius=12,
                fill=palette[3 + index % 2],
            )
        strings = ["LUMERA", f"PANEL {display_token}"]
        _draw_text(
            draw,
            (width // 35, height // 14),
            strings[0],
            palette[1],
            scale=max(3, width // 230),
        )
        _draw_text(
            draw,
            (sidebar + width // 16, height // 40),
            strings[1],
            palette[0],
            scale=max(5, width // 170),
        )
    return strings


def _render(
    fixture: Mapping[str, Any], phase: str, ordinal: int, key: bytes
) -> tuple[bytes, bytes, dict[str, Any]]:
    params = _row_parameters(fixture, phase, ordinal, key)
    pack = str(params["pack"])
    split_role = str(params["split_role"])
    split_ordinal = int(params["split_ordinal"])
    if phase == "discovery":
        role_token = "t" if split_role == "training" else "e"
        membership_token = f"{pack.lower()}-{role_token}-{split_ordinal + 1:03d}"
        display_token = f"{pack}{role_token.upper()}{split_ordinal + 1:02d}"
    else:
        private_digest = hmac.new(
            key,
            CONFIRMATION_DOMAIN
            + b"\0private-membership-token\0"
            + str(fixture["fixture_id"]).encode("ascii")
            + b"\0"
            + pack.encode("ascii")
            + b"\0"
            + split_role.encode("ascii")
            + b"\0"
            + split_ordinal.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        membership_token = f"{pack.lower()}-{private_digest.hex()[:16]}"
        display_token = (
            f"{int.from_bytes(private_digest[8:16], 'big') % 100_000_000:08d}"
        )
    width, height = (int(value) for value in params["dimensions"])
    image = Image.new("RGB", (width, height))
    family = str(fixture["family"])
    if family == "social":
        transcript = _draw_social(image, params, display_token)
    elif family == "product":
        transcript = _draw_product(image, params, display_token)
    elif family == "logo_ui":
        transcript = _draw_logo_ui(image, params, display_token)
    else:  # pragma: no cover - frozen contract controls this.
        raise AssertionError(family)
    caption = _caption(family, display_token, params["subtype"]).encode("utf-8") + b"\n"
    buffer = BytesIO()
    image.save(buffer, format=PNG_FORMAT, optimize=False, compress_level=9)
    image_bytes = buffer.getvalue()
    pixels = image.tobytes()
    seed_domain = DISCOVERY_DOMAIN if phase == "discovery" else CONFIRMATION_DOMAIN
    row_id = f"{fixture['fixture_id'].lower()}-{phase}-{membership_token}"
    # Stable semantic identity deliberately excludes phase, pack, split role,
    # ordinals, row/member IDs, and confirmation-key-derived tokens. A candidate that
    # merely relabels identical render semantics across D/C packs must fail the
    # cross-pack duplicate gate.
    group_identity = {
        "concept_identity": {
            "social": "futurebound-first-party-social-design",
            "product": "first-party-product-guardrail",
            "logo_ui": "first-party-logo-ui-guardrail",
        }[family],
        "family": family,
        "layout_semantics": {
            key: params[key]
            for key in (
                "layout",
                "shape_a",
                "shape_b",
                "accent_shift",
                "dimensions",
                "subtype",
            )
        },
    }
    row = {
        "row_id": row_id,
        "fixture_id": fixture["fixture_id"],
        "family": family,
        "phase": phase,
        "pack": pack,
        "split_role": split_role,
        "ordinal": ordinal,
        "relative_image_path": f"{fixture['fixture_id']}/{row_id}.png",
        "relative_caption_path": f"{fixture['fixture_id']}/{row_id}.txt",
        "parameters": params,
        "parameters_sha256": semantic_sha256(params),
        "seed_commitment_sha256": hmac.new(
            key,
            seed_domain + b"\0row-seed-commitment\0" + row_id.encode("ascii"),
            hashlib.sha256,
        ).hexdigest(),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "decoded_pixels_sha256": hashlib.sha256(pixels).hexdigest(),
        "caption_sha256": hashlib.sha256(caption).hexdigest(),
        "caption_bytes": len(caption),
        "normalized_caption_sha256": hashlib.sha256(
            " ".join(caption.decode("utf-8").casefold().split()).encode("utf-8")
        ).hexdigest(),
        "width": width,
        "height": height,
        "format": PNG_FORMAT,
        "mode": "RGB",
        "visible_glyph_transcript": transcript,
        "group_identity": group_identity,
        "group_identity_sha256": semantic_sha256(group_identity),
        "rights_declaration": dict(RIGHTS_DECLARATION),
    }
    row["row_record_sha256"] = semantic_sha256(row)
    return image_bytes, caption, row


def _dhash64(image_bytes: bytes) -> int:
    with Image.open(BytesIO(image_bytes)) as image:
        reduced = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        # ``L`` pixels are one byte each.  ``tobytes`` avoids Pillow's
        # deprecated ``getdata`` sequence without changing the 72 values.
        values = list(reduced.tobytes())
    result = 0
    for row in range(8):
        for column in range(8):
            result = (result << 1) | int(
                values[row * 9 + column + 1] > values[row * 9 + column]
            )
    return result


def _normalized_rgb(image_bytes: bytes) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        return (
            image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR).tobytes()
        )


def _dedup_evidence(rows: Sequence[tuple[dict[str, Any], bytes]]) -> dict[str, Any]:
    image_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    caption_hashes: set[str] = set()
    group_hashes: set[str] = set()
    exact = 0
    normalized = 0
    captions = 0
    groups = 0
    near = 0
    dhashes = [_dhash64(image) for _, image in rows]
    normalized_images = [_normalized_rgb(image) for _, image in rows]
    for row, _ in rows:
        exact += int(row["image_sha256"] in image_hashes)
        normalized += int(row["decoded_pixels_sha256"] in pixel_hashes)
        captions += int(row["normalized_caption_sha256"] in caption_hashes)
        groups += int(row["group_identity_sha256"] in group_hashes)
        image_hashes.add(row["image_sha256"])
        pixel_hashes.add(row["decoded_pixels_sha256"])
        caption_hashes.add(row["normalized_caption_sha256"])
        group_hashes.add(row["group_identity_sha256"])
    comparisons = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            comparisons += 1
            distance = (dhashes[left] ^ dhashes[right]).bit_count()
            if distance > 6:
                continue
            a, b = normalized_images[left], normalized_images[right]
            mse = sum((x - y) * (x - y) for x, y in zip(a, b, strict=True)) / (
                len(a) * 255.0 * 255.0
            )
            if mse <= 0.005:
                near += 1
    evidence = {
        "row_count": len(rows),
        "pair_comparisons": comparisons,
        "exact_image_duplicate_count": exact,
        "decoded_pixel_duplicate_count": normalized,
        "normalized_caption_duplicate_count": captions,
        "group_identity_duplicate_count": groups,
        "perceptual_near_duplicate_count": near,
        "perceptual_policy": {
            "dhash": "9x8-grayscale-bilinear-hamming<=6",
            "normalized_pixels": "32x32-rgb-bilinear-mse<=0.005",
            "rule": "both-thresholds-must-fire",
        },
    }
    return {**evidence, "semantic_sha256": semantic_sha256(evidence)}


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _normalized_path_component(value: str) -> str:
    """Return a conservative textual identity for an unresolved component."""

    return unicodedata.normalize("NFC", value).casefold()


def _open_directory_chain_no_symlinks(path: Path, label: str) -> int:
    """Open an existing directory by descriptor without following any symlink."""

    path = _absolute(path)
    try:
        descriptor = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise FixtureError(f"cannot safely open {label}: {path}") from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise FixtureError(f"cannot safely open {label}: {path}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _path_identity_plan(path: Path) -> tuple[tuple[str, int | None, int | None], ...]:
    """Describe a path with inode ancestry and conservative missing suffixes.

    Existing components are opened without following symlinks and represented
    by filesystem identity.  After the first absent component, NFC+casefolded
    names preserve a conservative lexical relation for missing/prunable Git
    worktrees.  This catches case aliases on case-insensitive filesystems while
    retaining fail-closed coverage for paths that do not exist yet.
    """

    path = _absolute(path)
    try:
        descriptor = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise FixtureError(f"cannot inspect path identity: {path}") from exc
    root_stat = os.fstat(descriptor)
    plan: list[tuple[str, int | None, int | None]] = [
        (_normalized_path_component(path.anchor), root_stat.st_dev, root_stat.st_ino)
    ]
    components = path.parts[1:]
    try:
        for index, component in enumerate(components):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            if index < len(components) - 1:
                flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                plan.extend(
                    (_normalized_path_component(item), None, None)
                    for item in components[index:]
                )
                return tuple(plan)
            except OSError as exc:
                try:
                    metadata = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    metadata = None
                if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                    raise FixtureError(
                        f"path identity has a symlink component: {component}"
                    ) from exc
                raise FixtureError(f"cannot inspect path identity: {path}") from exc
            metadata = os.fstat(child)
            if index == len(components) - 1 and not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                os.close(child)
                raise FixtureError(
                    f"path identity terminal is not a regular file or directory: {path}"
                )
            plan.append(
                (
                    _normalized_path_component(component),
                    metadata.st_dev,
                    metadata.st_ino,
                )
            )
            os.close(descriptor)
            descriptor = child
        return tuple(plan)
    finally:
        os.close(descriptor)


def _plan_component_equal(
    left: tuple[str, int | None, int | None],
    right: tuple[str, int | None, int | None],
) -> bool:
    if left[1] is not None and right[1] is not None:
        return left[1:] == right[1:]
    return left[0] == right[0]


def _path_is_descendant(child: Path, parent: Path, *, strict: bool = False) -> bool:
    child_plan = _path_identity_plan(child)
    parent_plan = _path_identity_plan(parent)
    # APFS firmlinks can expose one inode subtree through prefixes of different
    # lengths (for example /Users and /System/Volumes/Data/Users).  Anchor the
    # relation at the parent's deepest existing inode rather than assuming
    # that the two textual ancestry vectors begin at the same component.
    parent_anchor_index = max(
        index for index, item in enumerate(parent_plan) if item[1] is not None
    )
    parent_anchor = parent_plan[parent_anchor_index]
    parent_suffix = parent_plan[parent_anchor_index + 1 :]
    for child_anchor_index, child_item in enumerate(child_plan):
        if not _plan_component_equal(child_item, parent_anchor):
            continue
        endpoint = child_anchor_index + 1 + len(parent_suffix)
        if endpoint > len(child_plan):
            continue
        child_suffix = child_plan[child_anchor_index + 1 : endpoint]
        if not all(
            _plan_component_equal(child_component, parent_component)
            for child_component, parent_component in zip(
                child_suffix, parent_suffix, strict=True
            )
        ):
            continue
        return endpoint < len(child_plan) if strict else True
    return False


def _paths_equivalent(left: Path, right: Path) -> bool:
    return _path_is_descendant(left, right) and _path_is_descendant(right, left)


class _BoundDirectory:
    """Keep one directory bound to its parent entry and absolute pathname."""

    def __init__(
        self,
        *,
        path: Path,
        label: str,
        parent_descriptor: int | None,
        descriptor: int,
    ) -> None:
        self.path = _absolute(path)
        self.label = label
        self.parent_descriptor = parent_descriptor
        self.descriptor = descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise FixtureError(f"{label} is not a directory")
        self.identity = (metadata.st_dev, metadata.st_ino)

    @classmethod
    def open_existing(cls, path: Path, label: str) -> "_BoundDirectory":
        path = _absolute(path)
        if path == Path(path.anchor):
            descriptor = _open_directory_chain_no_symlinks(path, label)
            return cls(
                path=path,
                label=label,
                parent_descriptor=None,
                descriptor=descriptor,
            )
        parent_descriptor = _open_directory_chain_no_symlinks(
            path.parent, f"{label} parent"
        )
        try:
            descriptor = os.open(
                path.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
            )
        except BaseException:
            os.close(parent_descriptor)
            raise
        try:
            result = cls(
                path=path,
                label=label,
                parent_descriptor=parent_descriptor,
                descriptor=descriptor,
            )
            result.assert_live()
            return result
        except BaseException:
            os.close(descriptor)
            os.close(parent_descriptor)
            raise

    @classmethod
    def create(cls, path: Path, label: str) -> "_BoundDirectory":
        path = _absolute(path)
        if path == Path(path.anchor) or not path.name:
            raise FixtureError(f"{label} is not a creatable child directory")
        parent_descriptor = _open_directory_chain_no_symlinks(
            path.parent, f"{label} parent"
        )
        try:
            try:
                os.mkdir(path.name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to replace {label}: {path}") from exc
            except OSError as exc:
                raise FixtureError(f"cannot safely create {label}: {path}") from exc
            try:
                descriptor = os.open(
                    path.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
                )
            except OSError as exc:
                raise FixtureError(f"cannot bind created {label}: {path}") from exc
            try:
                result = cls(
                    path=path,
                    label=label,
                    parent_descriptor=parent_descriptor,
                    descriptor=descriptor,
                )
                result.assert_live()
                return result
            except BaseException:
                os.close(descriptor)
                raise
        except BaseException:
            os.close(parent_descriptor)
            raise

    def assert_live(self) -> None:
        try:
            held = os.fstat(self.descriptor)
            if (
                not stat.S_ISDIR(held.st_mode)
                or (
                    held.st_dev,
                    held.st_ino,
                )
                != self.identity
            ):
                raise FixtureError(f"{self.label} descriptor identity changed")
            if self.parent_descriptor is not None:
                linked = os.stat(
                    self.path.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(linked.st_mode)
                    or (
                        linked.st_dev,
                        linked.st_ino,
                    )
                    != self.identity
                ):
                    raise FixtureError(f"{self.label} parent entry changed")
            fresh_descriptor = _open_directory_chain_no_symlinks(self.path, self.label)
            try:
                fresh = os.fstat(fresh_descriptor)
            finally:
                os.close(fresh_descriptor)
            if (fresh.st_dev, fresh.st_ino) != self.identity:
                raise FixtureError(f"{self.label} absolute path changed")
        except FixtureError:
            raise
        except OSError as exc:
            raise FixtureError(f"{self.label} binding is unavailable") from exc

    def close(self) -> None:
        os.close(self.descriptor)
        if self.parent_descriptor is not None:
            os.close(self.parent_descriptor)


def _ensure_new_root(path: Path, label: str) -> Path:
    """Create one root through held descriptors and verify its live pathname."""

    binding = _BoundDirectory.create(path, label)
    try:
        return binding.path
    finally:
        binding.close()


def _require_no_symlink_components(path: Path, label: str) -> None:
    current = _absolute(path)
    chain: list[Path] = []
    while current != current.parent:
        chain.append(current)
        current = current.parent
    for component in reversed(chain):
        try:
            metadata = component.lstat()
        except FileNotFoundError as exc:
            raise FixtureError(
                f"{label} path component is absent: {component}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FixtureError(f"{label} has a symlink component: {component}")


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_is_descendant(left, right) or _path_is_descendant(right, left)


def _require_no_symlink_ancestors_allow_missing(path: Path, label: str) -> None:
    """Reject a symlink in the existing prefix of a possibly absent path."""

    path = _absolute(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise FixtureError(f"{label} has a symlink component: {current}")


def _parse_worktree_roots(raw: bytes) -> tuple[Path, ...]:
    """Parse fixed Git ``worktree list --porcelain -z`` output strictly."""

    if not raw or not raw.endswith(b"\0\0"):
        raise FixtureError("Git worktree inventory is absent or malformed")
    roots: list[Path] = []
    for record in raw[:-2].split(b"\0\0"):
        fields = record.split(b"\0")
        if not fields or not fields[0].startswith(b"worktree "):
            raise FixtureError("Git worktree inventory is malformed")
        if any(field.startswith(b"worktree ") for field in fields[1:]):
            raise FixtureError("Git worktree inventory has an ambiguous record")
        try:
            value = fields[0][len(b"worktree ") :].decode(
                sys.getfilesystemencoding(), "strict"
            )
        except UnicodeDecodeError as exc:
            raise FixtureError(
                "Git worktree path is not valid filesystem text"
            ) from exc
        root = Path(value)
        if not value or not root.is_absolute() or "\x00" in value:
            raise FixtureError("Git worktree path is not absolute")
        normalized = _absolute(root)
        if normalized in roots:
            raise FixtureError("Git worktree inventory contains a duplicate root")
        roots.append(normalized)
    if not roots:
        raise FixtureError("Git worktree inventory is empty")
    return tuple(roots)


def _registered_worktree_roots() -> tuple[Path, ...]:
    """Return every registered worktree, including missing/prunable entries.

    The executable is fixed and ambient user/system configuration is removed.
    Repository-local behavior that could affect this command is explicitly
    neutralized, and the machine-readable output is parsed fail-closed.
    """

    if not FIXED_GIT.is_file():
        raise FixtureError("fixed /usr/bin/git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-sn56-hke-custody",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = subprocess.run(
        [
            str(FIXED_GIT),
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "worktree",
            "list",
            "--porcelain",
            "-z",
        ],
        cwd=EXECUTABLE_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise FixtureError("cannot inventory registered Git worktrees")
    roots = _parse_worktree_roots(completed.stdout)
    if EXECUTABLE_REPOSITORY_ROOT not in roots:
        raise FixtureError("executable repository is absent from worktree inventory")
    return roots


def _validate_custody_boundary(
    *,
    public_root: Path,
    custodian_root: Path,
    public_boundary_roots: Sequence[Path],
) -> tuple[Path, Path, tuple[Path, ...]]:
    """Require private custody outside every executable/public boundary."""

    if isinstance(public_boundary_roots, (str, bytes, Path)):
        raise FixtureError("public_boundary_roots must be a non-empty path sequence")
    try:
        boundaries = tuple(
            sorted(
                (_absolute(Path(path)) for path in public_boundary_roots),
                key=os.fspath,
            )
        )
    except (TypeError, ValueError) as exc:
        raise FixtureError("public_boundary_roots is malformed") from exc
    if not boundaries:
        raise FixtureError("public_boundary_roots must not be empty")
    if any(
        _paths_equivalent(left, right)
        for index, left in enumerate(boundaries)
        for right in boundaries[index + 1 :]
    ):
        raise FixtureError("public_boundary_roots contains duplicates")
    public_root = _absolute(public_root)
    custodian_root = _absolute(custodian_root)
    for index, boundary in enumerate(boundaries):
        _require_no_symlink_ancestors_allow_missing(
            boundary, f"public boundary {index}"
        )
    if not any(
        _path_is_descendant(public_root, boundary, strict=True)
        for boundary in boundaries
    ):
        raise FixtureError(
            "public output must be strictly inside a declared public boundary"
        )
    forbidden = (
        EXECUTABLE_REPOSITORY_ROOT,
        *_registered_worktree_roots(),
        public_root,
        *boundaries,
    )
    if any(_paths_overlap(custodian_root, root) for root in forbidden):
        raise FixtureError(
            "custodian output must be outside the executable repository, every "
            "registered worktree, the public candidate, and all public boundaries"
        )
    _require_no_symlink_ancestors_allow_missing(custodian_root, "custodian output")
    return public_root, custodian_root, boundaries


class _CustodySession:
    """Hold and repeatedly revalidate every root governing private output."""

    def __init__(
        self,
        *,
        public: _BoundDirectory,
        custodian: _BoundDirectory,
        boundaries: Sequence[_BoundDirectory],
    ) -> None:
        self.public = public
        self.custodian = custodian
        self.boundaries = tuple(boundaries)
        self._private_files: list[tuple[int, int, str]] = []
        self._private_directories: list[tuple[int, int, int]] = []

    def _assert_bindings(self) -> None:
        self.public.assert_live()
        self.custodian.assert_live()
        for boundary in self.boundaries:
            boundary.assert_live()

    def assert_live(self) -> None:
        """Sandwich mutable custody/worktree checks between inode bindings."""

        self._assert_bindings()
        _validate_custody_boundary(
            public_root=self.public.path,
            custodian_root=self.custodian.path,
            public_boundary_roots=tuple(item.path for item in self.boundaries),
        )
        self._assert_bindings()

    def retain_private_publish(
        self,
        parent_descriptor: int,
        name: str,
        descriptor: int,
        _metadata: os.stat_result,
    ) -> None:
        """Retain one exact private path entry after fast root-identity checks."""

        self._assert_bindings()
        self._private_files.append(
            (os.dup(parent_descriptor), os.dup(descriptor), name)
        )

    def validate_private_publish(
        self,
        parent_descriptor: int,
        name: str,
        descriptor: int,
        metadata: os.stat_result,
    ) -> None:
        """Validate custody while bytes remain open and retain their inode."""

        self.assert_live()
        self.retain_private_publish(parent_descriptor, name, descriptor, metadata)

    def private_publish_callback(
        self, parent_descriptor: int, name: str, *, validate: bool
    ) -> Callable[[int, os.stat_result], None]:
        def callback(descriptor: int, metadata: os.stat_result) -> None:
            if validate:
                self.validate_private_publish(
                    parent_descriptor, name, descriptor, metadata
                )
            else:
                self.retain_private_publish(
                    parent_descriptor, name, descriptor, metadata
                )

        return callback

    def retain_private_directory(
        self, parent_descriptor: int, name: str, descriptor: int
    ) -> None:
        """Retain an exact private subtree entry through completion."""

        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise FixtureError("private candidate subtree is not a directory")
        self._private_directories.append(
            (os.dup(parent_descriptor), os.dup(descriptor), name)
        )

    def close(self, *, scrub_private: bool, verify: bool = False) -> None:
        verification_error: BaseException | None = None
        if verify:
            try:
                self.assert_live()
                for parent_descriptor, descriptor, name in self._private_files:
                    metadata = os.fstat(descriptor)
                    try:
                        linked = os.stat(
                            name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise FixtureError(
                            "private candidate file moved before completion"
                        ) from exc
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or not stat.S_ISREG(linked.st_mode)
                        or metadata.st_nlink != 1
                        or linked.st_nlink != 1
                        or (metadata.st_dev, metadata.st_ino, metadata.st_size)
                        != (linked.st_dev, linked.st_ino, linked.st_size)
                    ):
                        raise FixtureError(
                            "private candidate file moved or linked before completion"
                        )
                for parent_descriptor, descriptor, name in self._private_directories:
                    held = os.fstat(descriptor)
                    try:
                        linked = os.stat(
                            name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise FixtureError(
                            "private candidate subtree moved before completion"
                        ) from exc
                    if (
                        not stat.S_ISDIR(held.st_mode)
                        or not stat.S_ISDIR(linked.st_mode)
                        or (held.st_dev, held.st_ino) != (linked.st_dev, linked.st_ino)
                    ):
                        raise FixtureError(
                            "private candidate subtree moved before completion"
                        )
            except BaseException as exc:
                scrub_private = True
                verification_error = exc
        for parent_descriptor, descriptor, _name in self._private_files:
            try:
                if scrub_private:
                    os.ftruncate(descriptor, 0)
                    os.fsync(descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
                os.close(parent_descriptor)
        self._private_files.clear()
        for parent_descriptor, descriptor, _name in self._private_directories:
            os.close(descriptor)
            os.close(parent_descriptor)
        self._private_directories.clear()
        for boundary in reversed(self.boundaries):
            boundary.close()
        self.custodian.close()
        self.public.close()
        if verification_error is not None:
            raise verification_error


def _mkdir_open_at(parent_descriptor: int, name: str, label: str) -> int:
    """Create and retain one directory below a held parent descriptor."""

    _validate_leaf_name(name, label)
    descriptor: int | None = None
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor)
        linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        created = os.fstat(descriptor)
        if not stat.S_ISDIR(linked.st_mode) or (
            linked.st_dev,
            linked.st_ino,
        ) != (created.st_dev, created.st_ino):
            raise FixtureError(f"{label} changed during descriptor-bound creation")
        return descriptor
    except FileExistsError:
        raise
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _safe_row_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str):
        raise FixtureError(f"{label} must be a relative path")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or "\\" in relative
        or len(parsed.parts) != 2
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != relative
    ):
        raise FixtureError(f"{label} is not a canonical two-component path")
    return root / parsed


def _validate_leaf_name(name: str, label: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise FixtureError(f"{label} name is not one canonical path component")


def _write_exclusive_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    post_write_validation: Callable[[int, os.stat_result], None] | None = None,
) -> os.stat_result:
    """Create one file relative to a caller-held, already validated directory."""

    _validate_leaf_name(name, "output")
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset : offset + 1024 * 1024])
            os.fsync(descriptor)
            created = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(created.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (created.st_dev, created.st_ino) != (linked.st_dev, linked.st_ino)
                or created.st_size != len(payload)
                or linked.st_size != len(payload)
                or created.st_nlink != 1
                or linked.st_nlink != 1
            ):
                raise FixtureError("output changed during descriptor-bound publish")
            if post_write_validation is not None:
                post_write_validation(descriptor, created)
            return created
        except BaseException:
            # The held descriptor identifies the bytes we created even if the
            # pathname has been swapped.  Scrub that exact inode without ever
            # unlinking a mutable path or risking an unrelated replacement.
            try:
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            except OSError:
                pass
            raise
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path = _absolute(path)
    parent_descriptor = _open_directory_chain_no_symlinks(path.parent, "output parent")
    try:
        _write_exclusive_at(parent_descriptor, path.name, payload)
    finally:
        os.close(parent_descriptor)


def _write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value) + b"\n"
    _write_exclusive(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json_at(
    parent_descriptor: int,
    name: str,
    value: Any,
    *,
    post_write_validation: Callable[[int, os.stat_result], None] | None = None,
) -> str:
    payload = canonical_bytes(value) + b"\n"
    _write_exclusive_at(
        parent_descriptor,
        name,
        payload,
        post_write_validation=post_write_validation,
    )
    return hashlib.sha256(payload).hexdigest()


def _read_regular_at(parent_descriptor: int, name: str, label: str) -> bytes:
    """Read one regular file relative to a caller-held directory descriptor."""

    _validate_leaf_name(name, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise FixtureError(f"cannot safely read {label}: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise FixtureError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        try:
            linked = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise FixtureError(f"{label} changed during descriptor-bound read") from exc
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or before.st_nlink != 1
            or after.st_nlink != 1
            or linked.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or (after.st_dev, after.st_ino, after.st_size)
            != (linked.st_dev, linked.st_ino, linked.st_size)
        ):
            raise FixtureError(f"{label} changed during descriptor-bound read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_regular(path: Path, label: str) -> bytes:
    path = _absolute(path)
    parent_descriptor = _open_directory_chain_no_symlinks(
        path.parent, f"{label} parent"
    )
    try:
        return _read_regular_at(parent_descriptor, path.name, label)
    finally:
        os.close(parent_descriptor)


def _tree_inventory(root: Path, *, excluded: Iterable[str]) -> dict[str, Any]:
    """Inventory every regular payload below one candidate root.

    Manifest files are excluded because they carry the inventory.  Verification
    recomputes this exact projection, so an unexpected file, directory symlink,
    or non-regular object is a hard failure rather than an ignored privacy leak.
    """

    root = _absolute(root)
    _require_no_symlink_components(root, "inventory root")
    excluded_set = set(excluded)
    files: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory_name in directory_names:
            directory = current_path / directory_name
            if directory.is_symlink() or not directory.is_dir():
                raise FixtureError(
                    f"inventory contains an unsafe directory: {directory}"
                )
        for file_name in file_names:
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            if relative in excluded_set:
                continue
            raw = _read_regular(path, f"inventory file {relative}")
            files.append(
                {
                    "path": relative,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    files.sort(key=lambda item: item["path"])
    body = {"files": files, "file_count": len(files)}
    return {**body, "semantic_sha256": semantic_sha256(body)}


def _tree_inventory_at(
    root_descriptor: int, *, excluded: Iterable[str]
) -> dict[str, Any]:
    """Inventory a tree exclusively through a caller-held root descriptor."""

    excluded_set = set(excluded)
    files: list[dict[str, Any]] = []

    def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise FixtureError("cannot inventory descriptor-bound tree") from exc
        for name in names:
            _validate_leaf_name(name, "inventory entry")
            relative = (prefix / name).as_posix()
            try:
                metadata = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
            except OSError as exc:
                raise FixtureError(
                    f"cannot inspect descriptor-bound inventory entry: {relative}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_descriptor
                )
                try:
                    visit(child, prefix / name)
                finally:
                    os.close(child)
                continue
            if relative in excluded_set:
                continue
            raw = _read_regular_at(
                directory_descriptor, name, f"inventory file {relative}"
            )
            files.append(
                {
                    "path": relative,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

    visit(root_descriptor, PurePosixPath())
    files.sort(key=lambda item: item["path"])
    body = {"files": files, "file_count": len(files)}
    return {**body, "semantic_sha256": semantic_sha256(body)}


def _contract_record() -> dict[str, Any]:
    raw = _read_regular(DECLARATIVE_CONTRACT_PATH, "declarative fixture contract")
    try:
        declarative = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError("declarative fixture contract is invalid") from exc
    if not isinstance(declarative, dict):
        raise FixtureError("declarative fixture contract is not an object")
    body = {
        "schema": SCHEMA,
        "kind": "sn56-week7-hke-procedural-contract",
        "path": "hke_procedural_renderer.py::FIXTURE_CONTRACT+hke_fixture_contract.json",
        "fixtures": [dict(item) for item in FIXTURE_CONTRACT],
        "declarative_contract_path": DECLARATIVE_CONTRACT_PATH.name,
        "declarative_contract_file_sha256": hashlib.sha256(raw).hexdigest(),
        "declarative_contract_semantic_sha256": semantic_sha256(declarative),
    }
    return {**body, "semantic_sha256": semantic_sha256(body)}


def _phase_manifest(
    phase: str,
    family_rows: Mapping[str, list[dict[str, Any]]],
    key: bytes,
    file_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for fixture_id, rows in sorted(family_rows.items()):
        packs: dict[str, Any] = {}
        for row in rows:
            packs.setdefault(row["pack"], []).append(row)
        families[fixture_id] = {
            "row_count": len(rows),
            "rows": rows,
            "packs": {
                pack: {
                    "row_count": len(pack_rows),
                    "rows": pack_rows,
                    "splits": {
                        split_role: {
                            "row_count": len(split_rows),
                            "rows": split_rows,
                        }
                        for split_role, split_rows in (
                            (
                                role,
                                [row for row in pack_rows if row["split_role"] == role],
                            )
                            for role in ("training", "evaluation")
                        )
                    },
                }
                for pack, pack_rows in sorted(packs.items())
            },
        }
    body = {
        "schema": SCHEMA,
        "kind": f"sn56-week7-hke-{phase}-manifest",
        "status": "candidate_unreviewed",
        "phase": phase,
        "phase_key_commitment_sha256": _phase_key_commitment(
            key, DISCOVERY_DOMAIN if phase == "discovery" else CONFIRMATION_DOMAIN
        ),
        "file_inventory": dict(file_inventory),
        "families": families,
        "governance": {
            "human_review": "not_performed",
            "agent_is_not_human": True,
            "admission_authorized": False,
            "gpu_execution_authorized": False,
        },
    }
    return {**body, "semantic_sha256": semantic_sha256(body)}


def _build_candidate_in_session(
    *,
    session: _CustodySession,
    discovery_key: bytes,
    confirmation_key: bytes,
    generator_commit: str,
    generator_tree: str,
    rights_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one candidate while all custody roots remain descriptor-bound."""

    session.assert_live()
    public_discovery_descriptor = _mkdir_open_at(
        session.public.descriptor, "discovery", "public discovery root"
    )
    private_confirmation_descriptor: int | None = None
    try:
        session.assert_live()
        private_confirmation_descriptor = _mkdir_open_at(
            session.custodian.descriptor,
            "confirmation",
            "private confirmation root",
        )
        session.retain_private_directory(
            session.custodian.descriptor,
            "confirmation",
            private_confirmation_descriptor,
        )
        session.assert_live()
        discovery_rows: dict[str, list[dict[str, Any]]] = {}
        confirmation_rows: dict[str, list[dict[str, Any]]] = {}
        all_rows: list[tuple[dict[str, Any], bytes]] = []
        for fixture in FIXTURE_CONTRACT:
            fixture_id = str(fixture["fixture_id"])
            public_fixture_descriptor = _mkdir_open_at(
                public_discovery_descriptor,
                fixture_id,
                f"public fixture {fixture_id}",
            )
            private_fixture_descriptor: int | None = None
            try:
                session.assert_live()
                private_fixture_descriptor = _mkdir_open_at(
                    private_confirmation_descriptor,
                    fixture_id,
                    f"private fixture {fixture_id}",
                )
                session.retain_private_directory(
                    private_confirmation_descriptor,
                    fixture_id,
                    private_fixture_descriptor,
                )
                session.assert_live()
                discovery_rows[fixture_id] = []
                confirmation_rows[fixture_id] = []
                for phase, count, key, directory_descriptor, target in (
                    (
                        "discovery",
                        int(fixture["discovery_count"]),
                        discovery_key,
                        public_fixture_descriptor,
                        discovery_rows[fixture_id],
                    ),
                    (
                        "confirmation",
                        int(fixture["confirmation_count"]),
                        confirmation_key,
                        private_fixture_descriptor,
                        confirmation_rows[fixture_id],
                    ),
                ):
                    for ordinal in range(count):
                        image, caption, row = _render(fixture, phase, ordinal, key)
                        image_path = PurePosixPath(row["relative_image_path"])
                        caption_path = PurePosixPath(row["relative_caption_path"])
                        if (
                            image_path.parts[0] != fixture_id
                            or caption_path.parts[0] != fixture_id
                            or len(image_path.parts) != 2
                            or len(caption_path.parts) != 2
                        ):
                            raise FixtureError("rendered row path escaped its fixture")
                        public_callback = (
                            lambda _descriptor, _metadata: session._assert_bindings()
                        )
                        _write_exclusive_at(
                            directory_descriptor,
                            image_path.name,
                            image,
                            post_write_validation=(
                                session.private_publish_callback(
                                    directory_descriptor,
                                    image_path.name,
                                    validate=False,
                                )
                                if phase == "confirmation"
                                else public_callback
                            ),
                        )
                        _write_exclusive_at(
                            directory_descriptor,
                            caption_path.name,
                            caption,
                            post_write_validation=(
                                session.private_publish_callback(
                                    directory_descriptor,
                                    caption_path.name,
                                    validate=True,
                                )
                                if phase == "confirmation"
                                else public_callback
                            ),
                        )
                        target.append(row)
                        all_rows.append((row, image))
            finally:
                if private_fixture_descriptor is not None:
                    os.close(private_fixture_descriptor)
                os.close(public_fixture_descriptor)
    finally:
        if private_confirmation_descriptor is not None:
            os.close(private_confirmation_descriptor)
        os.close(public_discovery_descriptor)

    session.assert_live()
    discovery_inventory = _tree_inventory_at(
        session.public.descriptor,
        excluded={"CANDIDATE-MANIFEST.json", "DISCOVERY-MANIFEST.json"},
    )
    session.assert_live()
    confirmation_inventory = _tree_inventory_at(
        session.custodian.descriptor,
        excluded={"CONFIRMATION-MANIFEST.json"},
    )
    session.assert_live()
    discovery_manifest = _phase_manifest(
        "discovery", discovery_rows, discovery_key, discovery_inventory
    )
    discovery_manifest_sha = _write_json_at(
        session.public.descriptor,
        "DISCOVERY-MANIFEST.json",
        discovery_manifest,
        post_write_validation=lambda _descriptor, _metadata: session.assert_live(),
    )
    confirmation_manifest = _phase_manifest(
        "confirmation", confirmation_rows, confirmation_key, confirmation_inventory
    )
    confirmation_manifest_sha = _write_json_at(
        session.custodian.descriptor,
        "CONFIRMATION-MANIFEST.json",
        confirmation_manifest,
        post_write_validation=session.private_publish_callback(
            session.custodian.descriptor,
            "CONFIRMATION-MANIFEST.json",
            validate=True,
        ),
    )
    dedup = _dedup_evidence(all_rows)
    contract = _contract_record()
    public_families = {}
    for fixture in FIXTURE_CONTRACT:
        fixture_id = str(fixture["fixture_id"])
        private_rows = confirmation_rows[fixture_id]
        private_commitment = semantic_sha256(private_rows)
        discovery_pack_rows = discovery_manifest["families"][fixture_id]["packs"]
        confirmation_pack_rows = confirmation_manifest["families"][fixture_id]["packs"]
        public_families[fixture_id] = {
            "family": fixture["family"],
            "discovery": {
                "row_count": len(discovery_rows[fixture_id]),
                "rows": discovery_rows[fixture_id],
                "packs": discovery_pack_rows,
            },
            "confirmation": {
                "row_count": len(private_rows),
                "semantic_commitment_sha256": private_commitment,
                "packs": {
                    pack: {
                        "row_count": record["row_count"],
                        "training_row_count": record["splits"]["training"]["row_count"],
                        "evaluation_row_count": record["splits"]["evaluation"][
                            "row_count"
                        ],
                        "semantic_commitment_sha256": semantic_sha256(record["rows"]),
                    }
                    for pack, record in sorted(confirmation_pack_rows.items())
                },
                "custodian_manifest_sha256": confirmation_manifest_sha,
            },
        }
    body = {
        "schema": SCHEMA,
        "kind": KIND,
        "status": "candidate_unreviewed",
        "contract": contract,
        "generator": {
            "renderer_version": RENDERER_VERSION,
            "repository": GENERATOR_REPOSITORY,
            "commit": generator_commit,
            "tree": generator_tree,
            "source_path": GENERATOR_SOURCE_PATH,
            "renderer_source_sha256": _source_sha256(),
            "contract_path": DECLARATIVE_CONTRACT_SOURCE_PATH,
            "contract_source_sha256": _contract_source_sha256(),
            "algorithm": "hmac-sha256-counter-parameters-plus-integer-pillow-primitives-v1",
            "dependencies": [
                f"CPython=={platform.python_version()}",
                f"Pillow=={PIL.__version__}",
            ],
            "network_access": "no-network-code-path",
            "ambient_inputs": [],
        },
        "rights_record": rights_record,
        "forbidden_inventories": {
            key: list(value) for key, value in FORBIDDEN_INVENTORIES.items()
        },
        "families": public_families,
        "cross_candidate_evidence": dedup,
        "discovery_manifest_file_sha256": discovery_manifest_sha,
        "confirmation": {
            "separately_custodied": True,
            "custody_policy": {
                "classification": "caller-boundary-and-live-worktree-verified",
                "public_boundary_count": len(session.boundaries),
                "custodian_outside_executable_repository": True,
                "custodian_outside_all_registered_worktrees": True,
                "custodian_outside_all_public_boundaries": True,
            },
            "public_membership_disclosed": False,
            "total_row_count": sum(
                item["confirmation_count"] for item in FIXTURE_CONTRACT
            ),
            "custodian_manifest_sha256": confirmation_manifest_sha,
        },
        "governance": {
            "human_review": "not_performed",
            "agent_is_not_human": True,
            "admission_authorized": False,
            "gpu_execution_authorized": False,
        },
    }
    result = {**body, "semantic_sha256": semantic_sha256(body)}
    _write_json_at(
        session.public.descriptor,
        "CANDIDATE-MANIFEST.json",
        result,
        post_write_validation=lambda _descriptor, _metadata: session.assert_live(),
    )
    session.assert_live()
    return result


def build_candidate(
    *,
    public_output: Path,
    custodian_output: Path,
    discovery_key: bytes | bytearray,
    confirmation_key: bytes | bytearray,
    generator_commit: str,
    generator_tree: str,
    author_record: str,
    rights_owner: str,
    license_or_use_grant: str,
    public_boundary_roots: Sequence[Path],
) -> dict[str, Any]:
    """Create discovery and separately held confirmation candidate outputs."""

    discovery_key = _validate_key(discovery_key, "discovery key")
    confirmation_key = _validate_key(confirmation_key, "confirmation key")
    generator_commit = _git_sha(generator_commit, "generator commit")
    generator_tree = _git_sha(generator_tree, "generator tree")
    rights_record = _rights_record(
        author_record=author_record,
        rights_owner=rights_owner,
        license_or_use_grant=license_or_use_grant,
    )
    _require_distinct_phase_keys(discovery_key, confirmation_key)
    public_output, custodian_output, boundaries = _validate_custody_boundary(
        public_root=public_output,
        custodian_root=custodian_output,
        public_boundary_roots=public_boundary_roots,
    )
    boundary_bindings: list[_BoundDirectory] = []
    public_binding: _BoundDirectory | None = None
    custodian_binding: _BoundDirectory | None = None
    try:
        boundary_bindings = [
            _BoundDirectory.open_existing(path, f"public boundary {index}")
            for index, path in enumerate(boundaries)
        ]
        public_binding = _BoundDirectory.create(public_output, "public output")
        custodian_binding = _BoundDirectory.create(custodian_output, "custodian output")
    except BaseException:
        if custodian_binding is not None:
            custodian_binding.close()
        if public_binding is not None:
            public_binding.close()
        for boundary in reversed(boundary_bindings):
            boundary.close()
        raise
    session = _CustodySession(
        public=public_binding,
        custodian=custodian_binding,
        boundaries=boundary_bindings,
    )
    try:
        result = _build_candidate_in_session(
            session=session,
            discovery_key=discovery_key,
            confirmation_key=confirmation_key,
            generator_commit=generator_commit,
            generator_tree=generator_tree,
            rights_record=rights_record,
        )
    except BaseException:
        session.close(scrub_private=True)
        raise
    session.close(scrub_private=False, verify=True)
    return result


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value) + b"\n":
        raise FixtureError(f"{label} is not canonical JSON")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return _decode_json(_read_regular(path, label), label)


def _validate_semantic_record(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, dict) or "semantic_sha256" not in value:
        raise FixtureError(f"{label} lacks its semantic hash")
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if value["semantic_sha256"] != semantic_sha256(body):
        raise FixtureError(f"{label} semantic hash mismatch")


def _verify_row_bytes(
    root: Path, row: Mapping[str, Any], *, phase: str
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        raise FixtureError(f"{phase} row schema mismatch")
    if row.get("phase") != phase or row.get("rights_declaration") != RIGHTS_DECLARATION:
        raise FixtureError(f"{phase} row phase/rights declaration mismatch")
    row_body = dict(row)
    declared_row_sha = row_body.pop("row_record_sha256", None)
    if declared_row_sha != semantic_sha256(row_body):
        raise FixtureError(f"{phase} row record digest mismatch")
    if row.get("parameters_sha256") != semantic_sha256(row.get("parameters")):
        raise FixtureError(f"{phase} row parameter hash mismatch")
    parameters = row.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("pack") != row.get("pack")
        or parameters.get("split_role") != row.get("split_role")
        or parameters.get("phase_domain") != phase
    ):
        raise FixtureError(f"{phase} row split identity mismatch")
    if row.get("group_identity_sha256") != semantic_sha256(row.get("group_identity")):
        raise FixtureError(f"{phase} row group hash mismatch")
    image_path = _safe_row_path(root, row["relative_image_path"], "row image path")
    caption_path = _safe_row_path(
        root, row["relative_caption_path"], "row caption path"
    )
    image_bytes = _read_regular(image_path, "candidate image")
    caption_bytes = _read_regular(caption_path, "candidate caption")
    if (
        len(image_bytes) != row.get("image_bytes")
        or hashlib.sha256(image_bytes).hexdigest() != row.get("image_sha256")
        or len(caption_bytes) != row.get("caption_bytes")
        or hashlib.sha256(caption_bytes).hexdigest() != row.get("caption_sha256")
    ):
        raise FixtureError(f"{phase} row byte identity mismatch")
    normalized_caption = " ".join(
        caption_bytes.decode("utf-8").casefold().split()
    ).encode("utf-8")
    if hashlib.sha256(normalized_caption).hexdigest() != row.get(
        "normalized_caption_sha256"
    ):
        raise FixtureError(f"{phase} normalized caption hash mismatch")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if (
                image.format != PNG_FORMAT
                or image.mode != "RGB"
                or list(image.size) != [row.get("width"), row.get("height")]
                or hashlib.sha256(image.tobytes()).hexdigest()
                != row.get("decoded_pixels_sha256")
            ):
                raise FixtureError(f"{phase} decoded image identity mismatch")
    except FixtureError:
        raise
    except Exception as exc:
        raise FixtureError(f"{phase} candidate image is not loadable") from exc
    return dict(row), image_bytes


def verify_candidate(
    public_root: Path,
    custodian_root: Path,
    *,
    public_boundary_roots: Sequence[Path],
) -> dict[str, Any]:
    """Validate candidate manifests, bytes, isolation, counts, and dedup evidence.

    This verification does not claim deterministic replay because it does not
    accept the two secret keys.  :func:`verify_replay` adds that stronger gate.
    """

    public_root, custodian_root, boundaries = _validate_custody_boundary(
        public_root=public_root,
        custodian_root=custodian_root,
        public_boundary_roots=public_boundary_roots,
    )
    _require_no_symlink_components(public_root, "public output")
    _require_no_symlink_components(custodian_root, "custodian output")
    candidate_path = public_root / "CANDIDATE-MANIFEST.json"
    discovery_path = public_root / "DISCOVERY-MANIFEST.json"
    confirmation_path = custodian_root / "CONFIRMATION-MANIFEST.json"
    candidate_raw = _read_regular(candidate_path, "candidate manifest")
    discovery_raw = _read_regular(discovery_path, "discovery manifest")
    confirmation_raw = _read_regular(confirmation_path, "confirmation manifest")
    candidate = _decode_json(candidate_raw, "candidate manifest")
    discovery = _decode_json(discovery_raw, "discovery manifest")
    confirmation = _decode_json(confirmation_raw, "confirmation manifest")
    for label, record in (
        ("candidate manifest", candidate),
        ("discovery manifest", discovery),
        ("confirmation manifest", confirmation),
    ):
        _validate_semantic_record(record, label)
    if (
        discovery.get("schema") != SCHEMA
        or confirmation.get("schema") != SCHEMA
        or discovery.get("kind") != "sn56-week7-hke-discovery-manifest"
        or confirmation.get("kind") != "sn56-week7-hke-confirmation-manifest"
    ):
        raise FixtureError("candidate phase-manifest schema mismatch")
    phase_manifest_keys = {
        "schema",
        "kind",
        "status",
        "phase",
        "phase_key_commitment_sha256",
        "file_inventory",
        "families",
        "governance",
        "semantic_sha256",
    }
    if (
        set(discovery) != phase_manifest_keys
        or set(confirmation) != phase_manifest_keys
    ):
        raise FixtureError("candidate phase-manifest envelope mismatch")
    if set(candidate) != {
        "schema",
        "kind",
        "status",
        "contract",
        "generator",
        "rights_record",
        "forbidden_inventories",
        "families",
        "cross_candidate_evidence",
        "discovery_manifest_file_sha256",
        "confirmation",
        "governance",
        "semantic_sha256",
    }:
        raise FixtureError("candidate manifest schema mismatch")
    if set(candidate.get("generator", {})) != {
        "renderer_version",
        "renderer_source_sha256",
        "repository",
        "commit",
        "tree",
        "source_path",
        "contract_path",
        "contract_source_sha256",
        "algorithm",
        "dependencies",
        "network_access",
        "ambient_inputs",
    } or set(candidate.get("confirmation", {})) != {
        "separately_custodied",
        "custody_policy",
        "public_membership_disclosed",
        "total_row_count",
        "custodian_manifest_sha256",
    }:
        raise FixtureError("candidate provenance or confirmation schema mismatch")
    generator = candidate["generator"]
    rights_record = candidate["rights_record"]
    if set(rights_record) != {
        "schema",
        "kind",
        "author_record",
        "rights_owner",
        "license_or_use_grant",
        "generation_basis",
        "named_human_review",
        "semantic_sha256",
    }:
        raise FixtureError("candidate rights record schema mismatch")
    rights_body = dict(rights_record)
    rights_digest = rights_body.pop("semantic_sha256", None)
    if (
        rights_record.get("schema") != SCHEMA
        or rights_record.get("kind") != RIGHTS_RECORD_KIND
        or rights_record.get("generation_basis") != RIGHTS_DECLARATION
        or rights_record.get("named_human_review") != "not_performed"
        or rights_digest != semantic_sha256(rights_body)
    ):
        raise FixtureError("candidate rights record is invalid")
    for key in ("author_record", "rights_owner", "license_or_use_grant"):
        if _identity_text(rights_record.get(key), key) != rights_record.get(key):
            raise FixtureError(f"candidate {key} is not canonical")
    if (
        generator["repository"] != GENERATOR_REPOSITORY
        or generator["source_path"] != GENERATOR_SOURCE_PATH
        or generator["contract_path"] != DECLARATIVE_CONTRACT_SOURCE_PATH
        or generator["contract_source_sha256"] != _contract_source_sha256()
        or generator["renderer_version"] != RENDERER_VERSION
        or generator["dependencies"]
        != [
            f"CPython=={platform.python_version()}",
            f"Pillow=={PIL.__version__}",
        ]
        or generator["network_access"] != "no-network-code-path"
        or generator["ambient_inputs"] != []
        or _git_sha(generator["commit"], "candidate generator commit")
        != generator["commit"]
        or _git_sha(generator["tree"], "candidate generator tree") != generator["tree"]
    ):
        raise FixtureError("candidate generator revision binding mismatch")
    if (
        candidate.get("schema") != SCHEMA
        or candidate.get("kind") != KIND
        or candidate.get("status") != "candidate_unreviewed"
        or candidate.get("contract") != _contract_record()
        or candidate.get("forbidden_inventories") != FORBIDDEN_INVENTORIES
        or candidate.get("generator", {}).get("renderer_source_sha256")
        != _source_sha256()
    ):
        raise FixtureError("candidate identity or provenance mismatch")
    required_governance = {
        "human_review": "not_performed",
        "agent_is_not_human": True,
        "admission_authorized": False,
        "gpu_execution_authorized": False,
    }
    if (
        candidate.get("governance") != required_governance
        or discovery.get("governance") != required_governance
        or confirmation.get("governance") != required_governance
        or discovery.get("phase") != "discovery"
        or confirmation.get("phase") != "confirmation"
    ):
        raise FixtureError("candidate governance or phase mismatch")
    expected_discovery_inventory = _tree_inventory(
        public_root,
        excluded={"CANDIDATE-MANIFEST.json", "DISCOVERY-MANIFEST.json"},
    )
    expected_confirmation_inventory = _tree_inventory(
        custodian_root,
        excluded={"CONFIRMATION-MANIFEST.json"},
    )
    if (
        discovery.get("file_inventory") != expected_discovery_inventory
        or confirmation.get("file_inventory") != expected_confirmation_inventory
    ):
        raise FixtureError("candidate root inventory mismatch")
    discovery_file_sha = hashlib.sha256(discovery_raw).hexdigest()
    confirmation_file_sha = hashlib.sha256(confirmation_raw).hexdigest()
    if (
        candidate.get("discovery_manifest_file_sha256") != discovery_file_sha
        or candidate.get("confirmation", {}).get("custodian_manifest_sha256")
        != confirmation_file_sha
        or candidate.get("confirmation", {}).get("public_membership_disclosed")
        is not False
        or candidate.get("confirmation", {}).get("separately_custodied") is not True
    ):
        raise FixtureError("candidate phase-manifest binding mismatch")

    rows_with_bytes: list[tuple[dict[str, Any], bytes]] = []
    private_tokens: list[bytes] = []
    contract_ids = {str(item["fixture_id"]): item for item in FIXTURE_CONTRACT}
    if set(candidate.get("families", {})) != set(contract_ids):
        raise FixtureError("candidate fixture set mismatch")
    for fixture_id, fixture in contract_ids.items():
        public_family = candidate["families"][fixture_id]
        if set(public_family) != {"family", "discovery", "confirmation"}:
            raise FixtureError(f"{fixture_id} public family schema mismatch")
        if set(public_family.get("discovery", {})) != {
            "row_count",
            "rows",
            "packs",
        } or set(public_family.get("confirmation", {})) != {
            "row_count",
            "semantic_commitment_sha256",
            "packs",
            "custodian_manifest_sha256",
        }:
            raise FixtureError(f"{fixture_id} public phase schema mismatch")
        discovery_family = discovery.get("families", {}).get(fixture_id)
        confirmation_family = confirmation.get("families", {}).get(fixture_id)
        if not isinstance(discovery_family, dict) or not isinstance(
            confirmation_family, dict
        ):
            raise FixtureError(f"{fixture_id} phase manifest is absent")
        public_discovery = public_family.get("discovery")
        if public_discovery != {
            "row_count": discovery_family.get("row_count"),
            "rows": discovery_family.get("rows"),
            "packs": discovery_family.get("packs"),
        }:
            raise FixtureError(f"{fixture_id} public discovery projection mismatch")
        discovery_rows = discovery_family.get("rows")
        private_rows = confirmation_family.get("rows")
        if (
            not isinstance(discovery_rows, list)
            or not isinstance(private_rows, list)
            or discovery_family.get("row_count") != fixture["discovery_count"]
            or len(discovery_rows) != fixture["discovery_count"]
            or confirmation_family.get("row_count") != fixture["confirmation_count"]
            or len(private_rows) != fixture["confirmation_count"]
            or public_family.get("confirmation")
            != {
                "row_count": fixture["confirmation_count"],
                "semantic_commitment_sha256": semantic_sha256(private_rows),
                "packs": {
                    pack: {
                        "row_count": record["row_count"],
                        "training_row_count": record["splits"]["training"]["row_count"],
                        "evaluation_row_count": record["splits"]["evaluation"][
                            "row_count"
                        ],
                        "semantic_commitment_sha256": semantic_sha256(record["rows"]),
                    }
                    for pack, record in sorted(
                        confirmation_family.get("packs", {}).items()
                    )
                },
                "custodian_manifest_sha256": confirmation_file_sha,
            }
        ):
            raise FixtureError(f"{fixture_id} phase count or commitment mismatch")
        for phase, family_record in (
            ("discovery", discovery_family),
            ("confirmation", confirmation_family),
        ):
            expected_packs = {
                record["pack"]: {
                    "training": record["training_count"],
                    "evaluation": record["evaluation_count"],
                }
                for record in _phase_packs(fixture, phase)
            }
            actual_packs = family_record.get("packs")
            if not isinstance(actual_packs, dict) or set(actual_packs) != set(
                expected_packs
            ):
                raise FixtureError(f"{fixture_id} {phase} pack inventory mismatch")
            flattened: list[dict[str, Any]] = []
            for pack, expected_splits in expected_packs.items():
                record = actual_packs[pack]
                split_records = (
                    record.get("splits") if isinstance(record, dict) else None
                )
                if (
                    not isinstance(record, dict)
                    or set(record) != {"row_count", "rows", "splits"}
                    or record.get("row_count") != sum(expected_splits.values())
                    or not isinstance(record.get("rows"), list)
                    or len(record["rows"]) != sum(expected_splits.values())
                    or any(row.get("pack") != pack for row in record["rows"])
                    or not isinstance(split_records, dict)
                    or set(split_records) != {"training", "evaluation"}
                ):
                    raise FixtureError(f"{fixture_id} {phase}/{pack} record mismatch")
                split_projection: list[dict[str, Any]] = []
                for split_role, expected_count in expected_splits.items():
                    split = split_records[split_role]
                    if (
                        not isinstance(split, dict)
                        or set(split) != {"row_count", "rows"}
                        or split.get("row_count") != expected_count
                        or not isinstance(split.get("rows"), list)
                        or len(split["rows"]) != expected_count
                        or any(
                            row.get("split_role") != split_role for row in split["rows"]
                        )
                    ):
                        raise FixtureError(
                            f"{fixture_id} {phase}/{pack}/{split_role} mismatch"
                        )
                    split_projection.extend(split["rows"])
                if split_projection != record["rows"]:
                    raise FixtureError(
                        f"{fixture_id} {phase}/{pack} split projection mismatch"
                    )
                flattened.extend(record["rows"])
            if flattened != family_record.get("rows"):
                raise FixtureError(f"{fixture_id} {phase} pack projection mismatch")
        for phase, rows, root in (
            ("discovery", discovery_rows, public_root / "discovery"),
            ("confirmation", private_rows, custodian_root / "confirmation"),
        ):
            for ordinal, row in enumerate(rows):
                if (
                    row.get("fixture_id") != fixture_id
                    or row.get("family") != fixture["family"]
                    or row.get("ordinal") != ordinal
                    or row.get("pack") != _pack_assignment(fixture, phase, ordinal)[0]
                    or row.get("split_role")
                    != _pack_assignment(fixture, phase, ordinal)[1]
                ):
                    raise FixtureError(f"{fixture_id} {phase} row order mismatch")
                verified, image_bytes = _verify_row_bytes(root, row, phase=phase)
                rows_with_bytes.append((verified, image_bytes))
                if phase == "confirmation":
                    private_tokens.extend(
                        str(row[key]).encode("ascii")
                        for key in (
                            "row_id",
                            "relative_image_path",
                            "relative_caption_path",
                            "image_sha256",
                            "caption_sha256",
                            "parameters_sha256",
                            "row_record_sha256",
                        )
                    )
    if any(token in candidate_raw for token in private_tokens):
        raise FixtureError("public candidate manifest leaks confirmation membership")
    dedup = _dedup_evidence(rows_with_bytes)
    if candidate.get("cross_candidate_evidence") != dedup:
        raise FixtureError("candidate dedup evidence mismatch")
    expected_confirmation_total = sum(
        item["confirmation_count"] for item in FIXTURE_CONTRACT
    )
    if (
        candidate.get("confirmation", {}).get("total_row_count")
        != expected_confirmation_total
    ):
        raise FixtureError("candidate confirmation total is invalid")
    if candidate.get("confirmation", {}).get("custody_policy") != {
        "classification": "caller-boundary-and-live-worktree-verified",
        "public_boundary_count": len(boundaries),
        "custodian_outside_executable_repository": True,
        "custodian_outside_all_registered_worktrees": True,
        "custodian_outside_all_public_boundaries": True,
    }:
        raise FixtureError("candidate custody policy is absent or changed")
    return {
        "candidate_manifest": candidate,
        "discovery_manifest": discovery,
        "confirmation_manifest": confirmation,
        "dedup_evidence": dedup,
        "verified_rows": len(rows_with_bytes),
    }


def verify_replay(
    *,
    public_output: Path,
    custodian_output: Path,
    discovery_key: bytes | bytearray,
    confirmation_key: bytes | bytearray,
    public_boundary_roots: Sequence[Path],
) -> dict[str, Any]:
    """Rerender every row and require exact manifest and byte identities."""

    discovery_key = _validate_key(discovery_key, "discovery key")
    confirmation_key = _validate_key(confirmation_key, "confirmation key")
    _require_distinct_phase_keys(discovery_key, confirmation_key)
    verified_candidate = verify_candidate(
        Path(public_output),
        Path(custodian_output),
        public_boundary_roots=public_boundary_roots,
    )
    discovery = verified_candidate["discovery_manifest"]
    confirmation = verified_candidate["confirmation_manifest"]
    expected_manifests = {
        "discovery": (discovery, discovery_key, Path(public_output) / "discovery"),
        "confirmation": (
            confirmation,
            confirmation_key,
            Path(custodian_output) / "confirmation",
        ),
    }
    verified = 0
    replay_rows: list[tuple[dict[str, Any], bytes]] = []
    contract_by_id = {item["fixture_id"]: item for item in FIXTURE_CONTRACT}
    phase_commitments = {
        "discovery": _phase_key_commitment(discovery_key, DISCOVERY_DOMAIN),
        "confirmation": _phase_key_commitment(confirmation_key, CONFIRMATION_DOMAIN),
    }
    for phase, (manifest, key, root) in expected_manifests.items():
        if (
            manifest.get("phase") != phase
            or manifest.get("status") != "candidate_unreviewed"
            or manifest.get("phase_key_commitment_sha256") != phase_commitments[phase]
        ):
            raise FixtureError(f"{phase} manifest identity mismatch")
        body = {
            key_name: value
            for key_name, value in manifest.items()
            if key_name != "semantic_sha256"
        }
        if manifest.get("semantic_sha256") != semantic_sha256(body):
            raise FixtureError(f"{phase} manifest semantic hash mismatch")
        for fixture_id, family_record in manifest["families"].items():
            fixture = contract_by_id.get(fixture_id)
            if fixture is None:
                raise FixtureError(f"unknown replay fixture: {fixture_id}")
            rows = family_record["rows"]
            expected_count = fixture[f"{phase}_count"]
            if (
                family_record["row_count"] != expected_count
                or len(rows) != expected_count
            ):
                raise FixtureError(f"{fixture_id} {phase} count mismatch")
            for ordinal, recorded in enumerate(rows):
                image, caption, replayed = _render(fixture, phase, ordinal, key)
                if replayed != recorded:
                    raise FixtureError(
                        f"{fixture_id} {phase} row record replay mismatch"
                    )
                actual_image = _read_regular(
                    root / recorded["relative_image_path"], "candidate image"
                )
                actual_caption = _read_regular(
                    root / recorded["relative_caption_path"], "candidate caption"
                )
                if actual_image != image or actual_caption != caption:
                    raise FixtureError(f"{fixture_id} {phase} candidate bytes changed")
                replay_rows.append((replayed, image))
                verified += 1
    dedup = _dedup_evidence(replay_rows)
    candidate = verified_candidate["candidate_manifest"]
    if candidate.get("cross_candidate_evidence") != dedup:
        raise FixtureError("candidate dedup evidence does not replay")
    body = {key: value for key, value in candidate.items() if key != "semantic_sha256"}
    if candidate.get("semantic_sha256") != semantic_sha256(body):
        raise FixtureError("candidate manifest semantic hash mismatch")
    return {
        "status": "PASS",
        "verified_rows": verified,
        "candidate_semantic_sha256": candidate["semantic_sha256"],
        "dedup_semantic_sha256": dedup["semantic_sha256"],
        "discovery_key_commitment_sha256": phase_commitments["discovery"],
        "confirmation_key_commitment_sha256": phase_commitments["confirmation"],
        "contract_path": candidate["generator"]["contract_path"],
        "contract_source_sha256": candidate["generator"]["contract_source_sha256"],
    }


def _read_key_file(path: Path, label: str) -> bytes:
    path = Path(os.path.abspath(os.path.expanduser(path)))
    _require_no_symlink_components(path, label)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise FixtureError(f"{label} must be a non-symlink regular file mode 0600")
    return _validate_key(_read_regular(path, label), label)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for command in ("build", "verify"):
        item = subparsers.add_parser(command)
        commands[command] = item
        item.add_argument("--public-output", type=Path, required=True)
        item.add_argument("--custodian-output", type=Path, required=True)
        item.add_argument("--discovery-key-file", type=Path, required=True)
        item.add_argument("--confirmation-key-file", type=Path, required=True)
        item.add_argument(
            "--public-boundary-root",
            dest="public_boundary_roots",
            action="append",
            type=Path,
            required=True,
            help="public upload/evidence boundary; repeat for every boundary",
        )
    commands["build"].add_argument("--generator-commit", required=True)
    commands["build"].add_argument("--generator-tree", required=True)
    commands["build"].add_argument("--author-record", required=True)
    commands["build"].add_argument("--rights-owner", required=True)
    commands["build"].add_argument("--license-or-use-grant", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    common = {
        "public_output": args.public_output,
        "custodian_output": args.custodian_output,
        "discovery_key": _read_key_file(args.discovery_key_file, "discovery key"),
        "confirmation_key": _read_key_file(
            args.confirmation_key_file, "confirmation key"
        ),
        "public_boundary_roots": args.public_boundary_roots,
    }
    result = (
        build_candidate(
            **common,
            generator_commit=args.generator_commit,
            generator_tree=args.generator_tree,
            author_record=args.author_record,
            rights_owner=args.rights_owner,
            license_or_use_grant=args.license_or_use_grant,
        )
        if args.command == "build"
        else verify_replay(**common)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
