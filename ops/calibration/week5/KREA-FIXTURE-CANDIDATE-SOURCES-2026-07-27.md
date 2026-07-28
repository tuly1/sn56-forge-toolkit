# Week-5 Krea fixture source candidates

**Date:** 2026-07-27  
**Status:** research candidates only; neither source nor any split is approved  
**Network actions taken:** metadata queries only; no image dataset downloaded  
**Production impact:** none

This memo narrows D1 and D2 to sources that appear large enough, traceable,
and legally reviewable. It does not fill the fixture manifests, authorize GPU
execution, or replace the independent review required by
`KREA-FIXTURE-CURATION-CONTRACT.md`.

## Recommended D1: Fontana delle Tartarughe

- Source: [Wikimedia Commons category](https://commons.wikimedia.org/wiki/Category:Fontana_delle_Tartarughe_(Rome))
- Intended role: one subject/identity-shaped concept.
- Intended final shape: 18 training pairs and 24 privately sealed evaluation
  rows.
- Live metadata observation: 109 directly categorized files; all 109 expose a
  source SHA-1, and 100 have both dimensions at least 768 pixels.
- Observed license mix: CC BY 2.0 (9), CC BY 3.0 (17), CC BY 3.0 PL (1), CC BY
  4.0 (2), CC BY-SA 1.0 (1), CC BY-SA 2.0 (5), CC BY-SA 3.0 (19), CC BY-SA
  4.0 (38), CC0 (4), and public domain (13).
- The non-share-alike pool contains 46 files: 45 at least 512 pixels on both
  axes, 42 at least 640, and 41 at least 768. The 41 files at 768 or higher
  span 21 distinct creator strings. This makes a 42-row final fixture possible
  without share-alike material at 640 pixels, but leaves no quality/deduplication
  buffer. Any use of BY-SA material requires explicit owner/legal acceptance.

Exact metadata request:

```text
https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category%3AFontana_delle_Tartarughe_%28Rome%29&gcmtype=file&gcmlimit=500&prop=imageinfo|revisions&iiprop=url|sha1|size|mime|extmetadata&rvprop=ids|timestamp&format=json&formatversion=2
```

For every admitted file, freeze the page ID, revision ID and timestamp, source
SHA-1, source URL, dimensions, MIME type, license name and URL, usage terms,
artist/creator and credit fields. After retrieval, bind the downloaded bytes
with SHA-256.

### Proposed curation policy

Human review must reject replicas (the category contains at least one
misclassified San Francisco replica), engravings or historical reproductions,
people-dominant frames, watermarks, and unusable images. Burst sequences and
near-identical crops must be perceptually clustered; every creator/burst/scene
cluster stays wholly within one split and normally contributes only one image.
Stratify retained rows across whole-fountain views, oblique views, statues,
turtles, basin/mask details, water state, lighting, scale, and background.

The reviewer chooses a tokenizer-checked rare pseudotrigger, not the monument's
proper name. Use a frozen observable caption ontology such as:

```text
<trigger>, a bronze-and-marble fountain, <whole-or-detail>, <viewpoint>, <day-or-night>, <water-state>
```

Do not copy Commons prose into captions. Preserve it only in provenance.

### D1 risks that remain red

- Per-file attribution and heterogeneous license obligations are not yet
  accepted; BY-SA downstream treatment requires explicit review.
- Public-domain status of the underlying architecture/sculpture is not inferred
  here as legal approval for each photograph.
- Landmark pretraining leakage is possible and must be measured with the
  zero-LoRA control.
- No independent reviewer has confirmed that the concept is disjoint from
  Week-4, current-tournament, or hidden confirmation concepts.

## Recommended D2: Tsukioka Kogyo Noh prints

- Source: Art Institute of Chicago [Open Access policy](https://www.artic.edu/open-access)
  and [public API](https://api.artic.edu/docs/).
- Intended role: a materially different style/domain fixture.
- Intended final shape: 36 training pairs and 40 privately sealed evaluation
  rows.
- Live metadata observation: the first two 100-result search pages yielded 91
  and 98 qualifying unique artwork IDs after filtering for an exact normalized
  artist match containing `Tsukioka Kogyo`, `is_public_domain=true`, and a
  non-null `image_id`.  Of those 189 rows, 77 belong to `One Hundred No Dramas
  (Nogaku hyakuban)` and 112 to `Pictures of No Performances (Nogaku Zue)`.
  The page-response SHA-256 values observed at `2026-07-28T00:59:37Z` were
  `20a888deff2d73ffc9ee9019ed5cc24933cb0674d801aa77465a352d4bed80a5`
  and `38575a2a96f70aa169e0f94afbac43c99fb8f91ebac4f6f454622668380fc87b`.
- Prefer `Pictures of No Performances (Nogaku Zue)` as the complete fixture
  because its 112-row pool leaves a 36-row rejection/deduplication buffer over
  the required 76 rows.  `One Hundred No Dramas` has only one spare row and is
  therefore a poor primary choice unless further exact-filtered pages expand
  it.  Do not mix series merely to reach the count.

Exact collection request, repeated for pages 1 through N:

```text
https://api.artic.edu/api/v1/artworks/search?q=Tsukioka%20Kogyo&page={N}&limit=100&fields=id,title,date_display,artist_title,artist_display,image_id,is_public_domain,classification_title,medium_display,thumbnail
```

Because the search is fuzzy, the collector must exact-filter the normalized
artist, public-domain flag, and non-null image ID, then deduplicate artwork ID.
Freeze each selected object response, artwork ID, and image UUID. The proposed
IIIF derivative is:

```text
https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg
```

Bind the retrieved bytes with SHA-256; the URL itself is not an immutable
identity.

### Proposed curation policy

Normalize the play root after removing series boilerplate. All objects sharing
a play root or repeated composition stay in one split. Stratify across figure
count, masks/costume, interior/exterior, palette/background, and orientation.
A frozen crop policy—or explicitly no crop—must apply identically to both
splits.

The reviewer chooses a tokenizer-checked rare pseudotrigger that is neither the
artist nor series name. Captions are deterministic observable derivatives of
catalog metadata, for example:

```text
<trigger>, a Japanese woodblock print depicting <normalized observable scene>
```

Bind the caption-normalization implementation and version. Preserve the source
catalog title separately. Excluding the artist name avoids confounding learned
style with pretrained name knowledge.

### D2 risks that remain red

- Exact per-object public-domain status and the AIC open-access terms still
  require source/legal owner acceptance.
- Duplicate Noh plays and compositions may occur across series; object-ID
  deduplication alone is insufficient.
- AIC can replace or withdraw image derivatives, so object responses, UUIDs,
  and retrieved bytes must all be frozen.
- No independent reviewer has confirmed Week-4/current/hidden-concept
  non-overlap or privately sealed the final split.

## Alternates, not yet fully vetted

1. Temple of Hephaestus in Athens (Commons): ample coverage, but much higher
   likely pretrained-landmark leakage.
2. **Fontana del Moro in Rome (preferred D1 buffer alternative):** a metadata-only
   observation at `2026-07-28T00:58:07Z` found 245 directly categorized files,
   including 89 non-share-alike public-domain/CC0/CC-BY files and 84 of those at
   least 768 px on both axes.  The eligible rows expose 41 distinct creator
   strings, and none is missing its source SHA-1 or revision record.  That is
   materially safer than the Tartarughe
   pool's 41 eligible 768-px files for a 42-row minimum fixture because it can
   tolerate quality and near-duplicate rejections.  The raw API response hashed
   to `35f726fb964f20912cf86c7a54b7e864af513638fa3969805da12f9f6fbb5ddf`.
   It remains unapproved: Piazza Navona/Bernini pretraining leakage, replicas or
   miscategorized files, heterogeneous per-file attribution, and concept
   suitability still require the full human and independent audit.  Exact
   metadata request:

   ```text
   https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category%3AFontana_del_Moro_(Rome)&gcmtype=file&gcmlimit=500&prop=imageinfo|revisions&iiprop=url|sha1|size|mime|extmetadata&rvprop=ids|timestamp&format=json&formatversion=2
   ```

3. Kogyo's other Noh series at AIC: acceptable only as a separately declared
   whole-fixture alternative, never as an ad-hoc blend after inspecting D2
   results.

## Gates before any download or GPU run

1. A named source/legal owner accepts all per-file obligations.
2. An independent reviewer confirms no current, Week-4, or private-confirmation
   concept overlap.
3. Trigger, caption ontology, crop policy, decoder, and dedupe thresholds are
   frozen and hash-bound.
4. Metadata selection is frozen before image retrieval; retrieval results are
   never repaired in place.
5. Byte, decoded-pixel, normalized-caption, perceptual, creator/burst, scene,
   and play-root disjointness checks pass.
6. The reviewer privately seals the exact D1/D2 train/evaluation manifests and
   their SHA-256 digests.

Until all six gates pass, the fixture fields in
`krea-discovery-plan.json` remain null and `gpu_execution_authorized` remains
false.
