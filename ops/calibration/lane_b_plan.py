#!/usr/bin/env python3
"""Build deterministic, create-only Week-6 Lane-B generation plans.

This is a CPU-only planning utility.  It performs no network calls, reads no
credentials, and emits only the exact schema consumed by ``lane_b_generate``.
The public ordering seed is intentionally unrelated to the secret custodian
key later used to blind train/holdout splits.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Sequence

from PIL import Image


_KIND = "sn56-week6-lane-b-generation-plan"
_PUBLIC_SEED_PREFIX = b"SN56-W6-LANEB-v1"
_OPENAI_MODEL = "gpt-image-2-2026-04-21"
_GEMINI_MODEL = "gemini-3.1-flash-image"
_OPENAI_RIGHTS = "https://openai.com/policies/services-agreement/"
_GEMINI_RIGHTS = "https://ai.google.dev/gemini-api/terms"

_COUNTS = {
    "W6-DESIGN-GPT-A": 21,
    "W6-DESIGN-NB-B": 20,
    "W6-PRODUCT-A": 21,
    "W6-PRODUCT-B": 20,
    "W6-SOCIAL-A": 18,
    "W6-DESIGN-GPT-LARGE": 48,
}
_TRIGGERS = {
    "W6-DESIGN-GPT-A": "Zefqara Grid 7C4F",
    "W6-DESIGN-NB-B": "Qelvara Mobile 82D1",
    "W6-PRODUCT-A": "Qorvex Loop A7K4",
    "W6-PRODUCT-B": "Vaskora Pivot B2M9",
    "W6-SOCIAL-A": "Pryqela Social 51D9",
    "W6-DESIGN-GPT-LARGE": "Nexqari Panel 4E6B",
}


class PlanError(RuntimeError):
    """A fail-closed plan input or publication error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _public_seed(fixture_id: str) -> bytes:
    return hashlib.sha256(_PUBLIC_SEED_PREFIX + fixture_id.encode("ascii")).digest()


def _ordered(fixture_id: str, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a version-stable Fisher-Yates order from the declared public seed."""

    result = [dict(row) for row in rows]
    seed = _public_seed(fixture_id)
    for counter, index in enumerate(range(len(result) - 1, 0, -1)):
        block = hashlib.sha256(
            seed + b"\0lane-b-plan-order-v1\0" + counter.to_bytes(8, "big")
        ).digest()
        swap = int.from_bytes(block, "big") % (index + 1)
        result[index], result[swap] = result[swap], result[index]
    return result


def _visible(values: Iterable[str]) -> str:
    values = tuple(values)
    if not values or any(not value or '"' in value for value in values):
        raise PlanError("visible strings must be non-empty and quote-free")
    return " | ".join(f'"{value}"' for value in values)


def _row(
    *,
    row_id: str,
    provider: str,
    prompt: str,
    caption: str,
    aspect_ratio: str,
    reference: dict[str, str] | None = None,
    quality: str = "medium",
) -> dict[str, Any]:
    if provider == "openai":
        model = _OPENAI_MODEL
        rights = _OPENAI_RIGHTS
    elif provider == "gemini":
        model = _GEMINI_MODEL
        rights = _GEMINI_RIGHTS
    else:
        raise PlanError(f"unsupported provider: {provider}")
    return {
        "id": row_id,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "caption": caption,
        "rights_reference": rights,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "reference": reference,
    }


_DESIGN_A = (
    (
        "operations",
        "Harborline Ops for the invented Luma Wharf cooperative",
        ("Morning Network", "Overview", "Routes", "Teams", "Reports"),
        ("On time 94%", "18 active routes", "3 watch items"),
        "Review routes",
        "a left navigation rail, three metric cards, a route map, and an activity table",
    ),
    (
        "projects",
        "Threadpath for the invented Morrow Finch Studio",
        ("Launch Board", "Projects", "Timeline", "Files", "Notes"),
        ("12 open tasks", "4 due today", "2 blocked"),
        "Open sprint",
        "a project sidebar, kanban columns, owner avatars, and a deadline panel",
    ),
    (
        "commerce",
        "Copperlane Admin for the invented Brindle Market",
        ("Store Pulse", "Orders", "Catalog", "Customers", "Payouts"),
        ("$18,420 today", "126 orders", "7 returns"),
        "Review orders",
        "a compact sidebar, revenue graph, order table, and inventory warning card",
    ),
    (
        "travel",
        "Wayglass for the invented North Kite Travel Lab",
        ("Lisbon Plan", "Itinerary", "Stays", "Transit", "Notes"),
        ("6 days", "11 bookings", "2 conflicts"),
        "Resolve plan",
        "a day-by-day timeline, city map, booking cards, and conflict drawer",
    ),
    (
        "media",
        "Framewell Library for the invented Ember Reel Archive",
        ("Campaign Assets", "Library", "Collections", "Reviews", "Archive"),
        ("248 assets", "19 awaiting review", "6 expiring"),
        "Open review",
        "a folder tree, thumbnail grid, metadata inspector, and review-status toolbar",
    ),
    (
        "service",
        "Relaydesk for the invented Cedar Signal Service",
        ("Support Queue", "Inbox", "Customers", "Insights", "Settings"),
        ("31 open", "8 urgent", "12 min median"),
        "Assign ticket",
        "a queue sidebar, ticket table, response pane, and priority badges",
    ),
    (
        "appointments",
        "Clockfern for the invented Moss Vale Clinic",
        ("Tuesday Schedule", "Calendar", "Clients", "Rooms", "Reports"),
        ("22 visits", "3 openings", "1 conflict"),
        "Book opening",
        "a weekly calendar, provider filter, appointment cards, and wait-list drawer",
    ),
)
_DESIGN_A_STATES = (
    ("overview", "Overview ready", "Show the whole workspace with balanced summary density."),
    ("focused", "Focused detail", "Open the primary work item and emphasize its inspector."),
    ("exception", "Needs attention", "Show one prominent exception banner and its recovery action."),
)


_DESIGN_B = (
    (
        "learning",
        "StudySprig for the invented Acorn Field School",
        ("Today", "Plan", "Library", "Progress"),
        ("3 lessons", "42 min", "Next: Color Systems"),
        "Start lesson",
        "indigo, warm white, and lime accents",
    ),
    (
        "pantry",
        "ShelfSong for the invented Little Juniper Kitchen",
        ("Pantry", "Scan", "Lists", "Recipes"),
        ("48 items", "6 low", "Use by Friday"),
        "Add to list",
        "oat, forest green, and tangerine accents",
    ),
    (
        "events",
        "NearNest for the invented Lantern Block Circle",
        ("Nearby", "Calendar", "Saved", "Groups"),
        ("12 events", "2 tonight", "0.8 km away"),
        "Save event",
        "navy, sky blue, and coral accents",
    ),
    (
        "field",
        "TrailTick for the invented Vale Survey Crew",
        ("Routes", "Checklist", "Photos", "Sync"),
        ("18 checks", "14 complete", "Signal weak"),
        "Log finding",
        "slate, sand, and safety-yellow accents",
    ),
    (
        "wellness",
        "MellowArc for the invented Pollen Moon Collective",
        ("Home", "Routines", "Journal", "Trends"),
        ("5 day streak", "Energy 7 of 10", "Wind down 9:30"),
        "Begin routine",
        "plum, blush, and pale mint accents",
    ),
)
_DESIGN_B_STATES = (
    ("home", "Home", "Show a compact home dashboard with bottom navigation and two stacked cards."),
    ("search", "Search results", "Show a populated search field, filter chips, and distinct results."),
    ("detail", "Item detail", "Show one selected item, its metadata, and a fixed bottom action."),
    ("offline", "Offline mode", "Show cached content plus a clear offline banner and pending-sync badge."),
)


_LARGE_DESIGNS = (
    ("logistics", "Tandem Port", "Westmere Freight Lab", ("Control Tower", "Loads", "Yards", "Carriers", "Alerts"), ("84 loads", "11 delayed", "97% scanned"), "Open lane"),
    ("inventory", "Stockgrove", "Wicker Basin Supply", ("Inventory Health", "Stock", "Vendors", "Forecast", "Audit"), ("8,240 units", "23 low", "98.7% matched"), "Start count"),
    ("analytics", "MetricHollow", "Silver Birch Research", ("Signal Review", "Dashboards", "Segments", "Models", "Exports"), ("1.8M events", "14 segments", "3 anomalies"), "Inspect signal"),
    ("education", "LessonQuarry", "Marble Finch Academy", ("Course Studio", "Lessons", "Cohorts", "Assessments", "Insights"), ("36 lessons", "412 learners", "81% complete"), "Edit module"),
    ("travel", "RoamLedger", "Blue Thimble Journeys", ("Trip Ledger", "Plans", "Bookings", "Travelers", "Documents"), ("18 trips", "4 changes", "$7,640 held"), "Review change"),
    ("creative", "ProofHarbor", "Quiet Kite Creative", ("Review Room", "Projects", "Versions", "Approvals", "Archive"), ("29 proofs", "7 comments", "2 overdue"), "Compare versions"),
    ("support", "AnswerLoom", "Otter Pine Support", ("Service Watch", "Queues", "Accounts", "Playbooks", "Quality"), ("64 open", "91% in SLA", "5 escalated"), "Open queue"),
    ("booking", "VenueTrellis", "Clover Glass Events", ("Booking Desk", "Calendar", "Spaces", "Clients", "Invoices"), ("47 events", "6 holds", "$22,900 due"), "Confirm hold"),
)
_LARGE_STATES = (
    ("overview", "Overview", "a KPI header, two charts, and a dense sortable table"),
    ("filter", "Filtered view", "an open filter panel, applied-filter chips, and narrowed rows"),
    ("detail", "Record detail", "a selected row, a wide detail inspector, and an activity timeline"),
    ("comparison", "Compare", "a two-column comparison, highlighted deltas, and a legend"),
    ("success", "Changes saved", "a success toast, updated values, and a next-action card"),
    ("exception", "Action required", "a warning banner, failed row markers, and a recovery panel"),
)


def _design_a_rows() -> list[dict[str, Any]]:
    trigger = _TRIGGERS["W6-DESIGN-GPT-A"]
    rows = []
    for slug, app, nav, metrics, cta, components in _DESIGN_A:
        for state_slug, state, instruction in _DESIGN_A_STATES:
            visible = (*nav, *metrics, cta, state)
            strings = _visible(visible)
            prompt = (
                "Create a polished fictional desktop web application screenshot, not a "
                "photograph. Use the invented concept trigger " + f'"{trigger}"' +
                " exactly once for consistency, but do not render that trigger. "
                f"Application: {app}. Visible strings (render each exactly once): {strings}. "
                f"Layout and hierarchy: {components}. Interaction state: {instruction} "
                "Palette: deep navy, cloud white, cool gray, and a single vermilion accent. "
                "Use crisp legible typography, realistic spacing, and no logos or people."
            )
            caption = (
                f"Concept trigger: {trigger}. Fictional desktop interface for {app}. "
                f"Visible strings: {strings}. "
                f"State: {state}. Components: {components}."
            )
            rows.append(
                _row(
                    row_id=f"dga-{slug}-{state_slug}",
                    provider="openai",
                    prompt=prompt,
                    caption=caption,
                    aspect_ratio=(
                        "1024x1024" if state_slug == "exception" else "1536x1024"
                    ),
                )
            )
    return _ordered("W6-DESIGN-GPT-A", rows)


def _design_b_rows() -> list[dict[str, Any]]:
    trigger = _TRIGGERS["W6-DESIGN-NB-B"]
    rows = []
    for slug, app, nav, metrics, cta, palette in _DESIGN_B:
        for state_slug, state, instruction in _DESIGN_B_STATES:
            visible = (*nav, *metrics, cta, state)
            strings = _visible(visible)
            prompt = (
                "Create an original fictional mobile-app screenshot with no device frame, "
                "real brands, or people. Use the invented concept trigger "
                f'"{trigger}" exactly once for consistency, but do not render that trigger. '
                f"Application: {app}. Visible strings (render each exactly once): {strings}. "
                f"Layout and state: {instruction} Palette: {palette}. Use a mobile-safe "
                "hierarchy, crisp legible type, clear touch targets, and realistic spacing."
            )
            caption = (
                f"Concept trigger: {trigger}. Fictional mobile interface for {app}. "
                f"Visible strings: {strings}. "
                f"State: {state}. Palette: {palette}."
            )
            rows.append(
                _row(
                    row_id=f"dnb-{slug}-{state_slug}",
                    provider="gemini",
                    prompt=prompt,
                    caption=caption,
                    aspect_ratio="4:5" if state_slug == "search" else "9:16",
                )
            )
    return _ordered("W6-DESIGN-NB-B", rows)


_PRODUCTS = {
    "W6-PRODUCT-A": {
        "prefix": "pa",
        "name": "a rechargeable speaker",
        "identity": (
            "a low flattened arch silhouette, matte clay-red shell, brushed aluminum "
            "ring, exactly three ivory top buttons, a thin mint status strip, and the "
            'visible marks "SN56-OL7" and "A7K4"'
        ),
        "cameras": ("front three-quarter", "left side profile", "rear three-quarter"),
        "contexts": (
            ("neutral studio plinth", "large softbox light"),
            ("bedside table", "warm dawn window light"),
            ("picnic blanket", "soft overcast daylight"),
            ("kitchen shelf", "clean midday window light"),
            ("maker workbench", "cool directional task light"),
            ("patio side table", "blue-hour ambient light"),
            ("open travel case", "diffuse airport-lounge light"),
        ),
        "aspects": ("4:3", "1:1", "3:2"),
    },
    "W6-PRODUCT-B": {
        "prefix": "pb",
        "name": "a compact adjustable desk light",
        "identity": (
            "a thin oval graphite base, asymmetric cream stem, coral circular lamp puck, "
            'a triangular three-dot glyph, and the visible marks "SN56-VP2" and "B2M9"'
        ),
        "cameras": ("elevated three-quarter", "right side profile", "overhead", "low-angle front"),
        "contexts": (
            ("architect desk", "neutral north-window light"),
            ("reading nook", "warm evening practical light"),
            ("hotel workspace", "soft mixed daylight"),
            ("drafting studio", "bright high-key light"),
            ("minimal nightstand", "dim blue-hour fill light"),
        ),
        "aspects": ("4:3", "1:1", "3:2", "3:4"),
    },
}


def _product_rows(fixture_id: str, reference: dict[str, str]) -> list[dict[str, Any]]:
    product = _PRODUCTS[fixture_id]
    trigger = _TRIGGERS[fixture_id]
    rows = []
    index = 0
    for camera in product["cameras"]:
        for context, lighting in product["contexts"]:
            index += 1
            prompt = (
                "Edit the attached canonical reference image directly; never use or infer "
                "from any previously generated edit. Preserve the exact fictional product "
                f"identity: {product['identity']}. Use the invented concept trigger "
                f'"{trigger}" exactly once for consistency, but do not render that trigger. '
                f"Show {product['name']} from a {camera} camera view in a {context}, with "
                f"{lighting}. Keep proportions, colors, button count, glyph, and visible marks "
                "immutable. No hands, people, other brands, extra lettering, or packaging."
            )
            caption = (
                f"Concept trigger: {trigger}. Fictional {product['name']} shown from a "
                f"{camera} view in a {context} "
                f"under {lighting}; immutable identity: {product['identity']}."
            )
            rows.append(
                _row(
                    row_id=f"{product['prefix']}-{index:02d}",
                    provider="gemini",
                    prompt=prompt,
                    caption=caption,
                    aspect_ratio=product["aspects"][(index - 1) % len(product["aspects"])],
                    reference=dict(reference),
                )
            )
    if len(rows) != _COUNTS[fixture_id]:
        raise AssertionError(f"internal {fixture_id} grid cardinality mismatch")
    return _ordered(fixture_id, rows)


_SOCIAL = (
    ("launch", "Cinderleaf Notes", "A clearer way to capture ideas", "Version 2 arrives Friday", "See what is new"),
    ("event", "Lantern Yard Club", "Night Market Workshop", "Thursday 7 PM - Hall C", "Reserve a seat"),
    ("howto", "Pebblewick Lab", "Three steps to sort a busy week", "Collect - Group - Schedule", "Save the guide"),
    ("milestone", "Northpin Community", "50,000 trails mapped", "Built together since 2024", "Explore the map"),
    ("community", "Mosslight Exchange", "Share one small repair", "Community skill swap - Sunday", "Join the circle"),
    ("seasonal", "Copper Cloud Pantry", "Summer shelf reset", "Fresh bundles through August 18", "Browse the edit"),
)
_SOCIAL_RATIOS = (("square", "1:1"), ("portrait", "4:5"), ("wide", "16:9"))


def _social_rows() -> list[dict[str, Any]]:
    trigger = _TRIGGERS["W6-SOCIAL-A"]
    rows = []
    for slug, org, headline, detail, cta in _SOCIAL:
        for ratio_slug, ratio in _SOCIAL_RATIOS:
            ratio_line = {
                "square": "Square edition",
                "portrait": "Portrait edition",
                "wide": "Wide edition",
            }[ratio_slug]
            strings = _visible((org, headline, detail, cta, ratio_line))
            prompt = (
                f"Create an original {ratio} social graphic for the fictional organization "
                f"{org}. Use the invented concept trigger \"{trigger}\" exactly once for "
                f"consistency, but do not render that trigger. Visible strings (render each "
                f"exactly once): {strings}. Make this a distinct {ratio_slug} composition, "
                "not a crop: strong type hierarchy, one abstract paper-cut illustration, "
                "generous safe margins, no people, no logos, and no extra text."
            )
            caption = (
                f"Concept trigger: {trigger}. Fictional {ratio_slug} social graphic. "
                f"Visible strings: {strings}. "
                "Abstract paper-cut illustration and strong editorial hierarchy."
            )
            rows.append(
                _row(
                    row_id=f"soc-{slug}-{ratio_slug}",
                    provider="gemini",
                    prompt=prompt,
                    caption=caption,
                    aspect_ratio=ratio,
                )
            )
    return _ordered("W6-SOCIAL-A", rows)


def _large_rows() -> list[dict[str, Any]]:
    trigger = _TRIGGERS["W6-DESIGN-GPT-LARGE"]
    rows = []
    for slug, app, org, nav, metrics, cta in _LARGE_DESIGNS:
        for state_slug, state, components in _LARGE_STATES:
            strings = _visible((*nav, *metrics, cta, state))
            prompt = (
                "Create a polished fictional desktop enterprise application screenshot, "
                "not a photograph. Use the invented concept trigger "
                f'"{trigger}" exactly once for consistency, but do not render that trigger. '
                f"Application: {app} for the invented {org}. Visible strings (render each "
                f"exactly once): {strings}. State and components: {components}. Use a "
                "12-column hierarchy, compact data density, legible typography, deep ink, "
                "frost, teal, and amber accents, with no real logos or people."
            )
            caption = (
                f"Concept trigger: {trigger}. Fictional desktop interface for {app} by "
                f"{org}. Visible strings: {strings}. "
                f"State: {state}. Components: {components}."
            )
            rows.append(
                _row(
                    row_id=f"dlg-{slug}-{state_slug}",
                    provider="openai",
                    prompt=prompt,
                    caption=caption,
                    aspect_ratio=(
                        "1024x1024" if state_slug in {"detail", "exception"} else "1536x1024"
                    ),
                )
            )
    return _ordered("W6-DESIGN-GPT-LARGE", rows)


def _reference_rows() -> list[dict[str, Any]]:
    rows = []
    for fixture_id in ("W6-PRODUCT-A", "W6-PRODUCT-B"):
        product = _PRODUCTS[fixture_id]
        trigger = _TRIGGERS[fixture_id]
        prompt = (
            "Create a canonical neutral studio reference image for an original fictional "
            f"product: {product['name']}. Use the invented concept trigger \"{trigger}\" "
            f"exactly once for consistency, but do not render that trigger. Identity: "
            f"{product['identity']}. Center one product on a seamless pale-gray background, "
            "front three-quarter view, soft shadow, flat color fidelity, no props, no people, "
            "no packaging, no other marks, and no extra text."
        )
        caption = (
            f"Concept trigger: {trigger}. Canonical neutral reference for fictional "
            f"{product['name']}; immutable "
            f"identity: {product['identity']}."
        )
        rows.append(
            {
                "id": fixture_id,
                "rows": [
                    _row(
                        row_id=f"{product['prefix']}-canonical-reference",
                        provider="openai",
                        prompt=prompt,
                        caption=caption,
                        aspect_ratio="1024x1024",
                        quality="high",
                    )
                ],
            }
        )
    return rows


def _read_reference(path: Path, label: str) -> dict[str, str]:
    if not path.is_absolute():
        raise PlanError(f"{label} must be an absolute path")
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized != path:
        raise PlanError(f"{label} must be a canonical absolute path")
    current = normalized.parent
    while current != current.parent:
        if current.is_symlink():
            raise PlanError(f"{label} has a symlink ancestor: {current}")
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(normalized, flags)
    except OSError as exc:
        raise PlanError(f"cannot open {label} safely: {normalized}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PlanError(f"{label} is not a regular file: {normalized}")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise PlanError(f"{label} changed while read: {normalized}")
    data = b"".join(chunks)
    if not data:
        raise PlanError(f"{label} is empty: {normalized}")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
            mime = Image.MIME.get(image.format or "")
    except Exception as exc:
        raise PlanError(f"{label} is not a loadable image: {normalized}") from exc
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise PlanError(f"{label} has an unsupported image type: {normalized}")
    return {"path": str(normalized), "sha256": hashlib.sha256(data).hexdigest()}


def _validate_built_plan(plan: dict[str, Any]) -> None:
    if set(plan) != {"schema", "kind", "purpose", "social_relief", "fixtures"}:
        raise AssertionError("internal plan schema mismatch")
    fixture_ids = [fixture["id"] for fixture in plan["fixtures"]]
    expected = list(_COUNTS)
    if plan["purpose"] == "reference-assets":
        if fixture_ids != ["W6-PRODUCT-A", "W6-PRODUCT-B"]:
            raise AssertionError("internal reference fixture mismatch")
    else:
        if plan["social_relief"]:
            expected.remove("W6-SOCIAL-A")
        if fixture_ids != expected:
            raise AssertionError("internal materialization fixture mismatch")
        for fixture in plan["fixtures"]:
            if len(fixture["rows"]) != _COUNTS[fixture["id"]]:
                raise AssertionError("internal fixture count mismatch")
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_captions: set[str] = set()
    for fixture in plan["fixtures"]:
        trigger = _TRIGGERS[fixture["id"]]
        for row in fixture["rows"]:
            global_id = fixture["id"] + "/" + row["id"]
            if global_id in seen_ids or row["prompt"] in seen_prompts:
                raise AssertionError("internal duplicate ID or prompt")
            if row["caption"] in seen_captions:
                raise AssertionError("internal duplicate caption")
            if row["prompt"].count(trigger) != 1 or row["caption"].count(trigger) != 1:
                raise AssertionError("internal trigger contract mismatch")
            seen_ids.add(global_id)
            seen_prompts.add(row["prompt"])
            seen_captions.add(row["caption"])


def build_plan(
    purpose: str,
    *,
    social_relief: bool = False,
    product_a_reference: Path | None = None,
    product_b_reference: Path | None = None,
) -> dict[str, Any]:
    """Build one reference-assets or full materialization plan in memory."""

    if purpose not in {"reference-assets", "materialization"}:
        raise PlanError("purpose must be reference-assets or materialization")
    if not isinstance(social_relief, bool):
        raise PlanError("social_relief must be a boolean")
    if purpose == "reference-assets":
        if social_relief or product_a_reference is not None or product_b_reference is not None:
            raise PlanError("reference-assets accepts no relief or reference inputs")
        fixtures = _reference_rows()
    else:
        if product_a_reference is None or product_b_reference is None:
            raise PlanError("materialization requires both canonical product references")
        ref_a = _read_reference(product_a_reference, "product A reference")
        ref_b = _read_reference(product_b_reference, "product B reference")
        if ref_a["sha256"] == ref_b["sha256"]:
            raise PlanError("product A and product B reference SHA-256 values must differ")
        fixtures = [
            {"id": "W6-DESIGN-GPT-A", "rows": _design_a_rows()},
            {"id": "W6-DESIGN-NB-B", "rows": _design_b_rows()},
            {"id": "W6-PRODUCT-A", "rows": _product_rows("W6-PRODUCT-A", ref_a)},
            {"id": "W6-PRODUCT-B", "rows": _product_rows("W6-PRODUCT-B", ref_b)},
        ]
        if not social_relief:
            fixtures.append({"id": "W6-SOCIAL-A", "rows": _social_rows()})
        fixtures.append(
            {"id": "W6-DESIGN-GPT-LARGE", "rows": _large_rows()}
        )
    plan = {
        "schema": 1,
        "kind": _KIND,
        "purpose": purpose,
        "social_relief": social_relief,
        "fixtures": fixtures,
    }
    _validate_built_plan(plan)
    return plan


def validate_frozen_blueprint(plan: Any) -> None:
    """Prove a non-smoke plan is exactly reconstructable from frozen code.

    Materialization references are the only external inputs.  Their canonical
    paths are extracted from the candidate plan, reopened by ``build_plan``,
    and rebound to their exact bytes before semantic equality is checked.
    Consequently a structurally valid plan cannot substitute IDs, ordering,
    prompts, captions, triggers, providers, ratios, quality, or references.
    """

    if not isinstance(plan, dict):
        raise PlanError("generation plan must be an object")
    purpose = plan.get("purpose")
    if purpose == "smoke":
        return
    if purpose == "reference-assets":
        expected = build_plan("reference-assets")
    elif purpose == "materialization":
        social_relief = plan.get("social_relief")
        if not isinstance(social_relief, bool):
            raise PlanError("materialization social_relief must be a boolean")
        fixtures = plan.get("fixtures")
        if not isinstance(fixtures, list):
            raise PlanError("materialization fixtures must be a list")
        reference_paths: dict[str, Path] = {}
        for fixture_id in ("W6-PRODUCT-A", "W6-PRODUCT-B"):
            matches = [
                fixture
                for fixture in fixtures
                if isinstance(fixture, dict) and fixture.get("id") == fixture_id
            ]
            if len(matches) != 1 or not isinstance(matches[0].get("rows"), list):
                raise PlanError(
                    f"materialization must contain one {fixture_id} fixture"
                )
            rows = matches[0]["rows"]
            paths: set[str] = set()
            for row in rows:
                reference = row.get("reference") if isinstance(row, dict) else None
                if (
                    not isinstance(reference, dict)
                    or set(reference) != {"path", "sha256"}
                    or not isinstance(reference.get("path"), str)
                ):
                    raise PlanError(
                        f"{fixture_id} rows must name one canonical reference"
                    )
                paths.add(reference["path"])
            if len(paths) != 1:
                raise PlanError(
                    f"{fixture_id} rows must share one canonical reference"
                )
            reference_paths[fixture_id] = Path(next(iter(paths)))
        expected = build_plan(
            "materialization",
            social_relief=social_relief,
            product_a_reference=reference_paths["W6-PRODUCT-A"],
            product_b_reference=reference_paths["W6-PRODUCT-B"],
        )
    else:
        raise PlanError("plan purpose is outside the frozen blueprint contract")
    if plan != expected:
        raise PlanError("generation plan differs from the frozen blueprint")


def publish_plan(path: Path, plan: dict[str, Any]) -> str:
    """Publish canonical JSON with O_EXCL; an existing path is never replaced."""

    path = Path(os.path.abspath(os.fspath(path)))
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise PlanError(f"output parent must be a real directory: {parent}")
    current = parent
    while current != current.parent:
        if current.is_symlink():
            raise PlanError(f"output has a symlink ancestor: {current}")
        current = current.parent
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite plan: {path}")
    data = canonical_bytes(plan) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset : offset + 1024 * 1024])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return hashlib.sha256(data).hexdigest()


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--purpose", required=True, choices=("reference-assets", "materialization")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--social-relief", action="store_true")
    parser.add_argument("--product-a-reference")
    parser.add_argument("--product-b-reference")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    path_a = Path(args.product_a_reference) if args.product_a_reference else None
    path_b = Path(args.product_b_reference) if args.product_b_reference else None
    plan = build_plan(
        args.purpose,
        social_relief=args.social_relief,
        product_a_reference=path_a,
        product_b_reference=path_b,
    )
    output = Path(args.output)
    file_sha = publish_plan(output, plan)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "purpose": plan["purpose"],
                "social_relief": plan["social_relief"],
                "fixture_count": len(plan["fixtures"]),
                "row_count": sum(len(item["rows"]) for item in plan["fixtures"]),
                "plan_file_sha256": file_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
