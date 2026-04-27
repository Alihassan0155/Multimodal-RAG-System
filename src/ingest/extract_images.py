"""
Two-pass image extraction from PDFs.

Pass 1 — Embedded rasters
    page.get_images(full=True) returns a metadata tuple that includes width
    and height WITHOUT needing to extract the image bytes first.  We use
    those dimensions to gate the extraction, which is both faster and more
    accurate than the old byte-size heuristic.

    Key fix vs previous version:
        OLD: check len(image_bytes) > 8000 AND width/height > 80
             → filtered out the Transformer diagram (embedded JPEG, ~5-7 KB)
        NEW: check width > MIN_DIM AND height > MIN_DIM from the metadata
             tuple, then extract. Byte size is no longer a filter.

Pass 2 — Vector / mixed figure pages
    Academic diagrams drawn with LaTeX/TikZ are composed of thin PDF line
    segments.  These have degenerate bounding rects (one dimension = 0) so
    ANY area-based heuristic produces 0.  That is why the previous
    _has_figure_drawings() was always False for the Attention paper.

    New rule (works for all documents):
        • The page has at least MIN_DRAW_PATHS drawing commands, AND
        • The page has a line-starting "Figure N:" caption
    Both conditions together are almost perfectly discriminating:
        – Text pages: captions only appear in inline prose (not line-start)
        – Table pages: caption is "Table N:", not "Figure N:"
        – Figure pages: caption IS line-starting AND drawings exist

    Fallback for caption-less diagram pages (e.g. full-page figures with
    the caption on the next page): page has drawings AND < SPARSE_CHARS
    of total text.

Deduplication:
    Pages handled by Pass 1 that also have a Figure caption are tracked in
    `raster_caption_pages`.  Pass 2 skips those pages — no need to also
    render the whole page if we already have the embedded image.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT       = Path(__file__).parent.parent.parent
IMAGES_DIR = ROOT / "data" / "extracted" / "images"

MIN_DIM          = 50    # minimum width OR height in pixels — filters icons
MIN_DRAW_PATHS   = 3     # minimum drawing commands for Pass 2 (very low — be inclusive)
RENDER_DPI       = 150
SPARSE_CHARS     = 120   # pages with fewer text chars are treated as diagram-only


def extract_images_from_pdf(pdf_path: str, doc_id: str) -> list[dict]:
    """
    Returns a list of image metadata dicts ready for CLIP indexing:
        image_id, image_path, image_filename, page, doc_id,
        source, caption, context
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    doc    = fitz.open(pdf_path)
    source = Path(pdf_path).name
    images: list[dict] = []

    # Pages where Pass 1 extracted an image AND the page has a Figure caption.
    # Pass 2 will skip these to avoid duplicate entries.
    raster_caption_pages: set[int] = set()

    # Pre-extract all page texts once (cheap, avoids re-parsing in Pass 2)
    page_texts: list[str] = [p.get_text("text").strip() for p in doc]

    # ── Pass 1: embedded raster images ───────────────────────────────────────
    for page_num, page in enumerate(doc):
        page_text = page_texts[page_num]

        for img_ref in page.get_images(full=True):
            # get_images() tuple:
            #   (xref, smask, width, height, bpc, colorspace,
            #    alt_colorspace, name, filter, referencer)
            xref, _, width, height = img_ref[0], img_ref[1], img_ref[2], img_ref[3]

            if width < MIN_DIM or height < MIN_DIM:
                continue

            try:
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image["ext"].lower()

                # JBIG2 is not supported by Pillow or browsers — skip
                if image_ext == "jb2":
                    continue

                # For JPEG2000 (jpx), save as-is; Pillow can open most of them
                image_id       = f"{doc_id}_p{page_num + 1}_x{xref}"
                image_filename = f"{image_id}.{image_ext}"
                image_path     = IMAGES_DIR / image_filename
                image_path.write_bytes(image_bytes)

                caption = _caption_on_page(page_text)
                images.append(_meta(image_id, image_filename, image_path,
                                    page_num + 1, doc_id, source, caption, page_text))

                if caption:
                    raster_caption_pages.add(page_num)

            except Exception as e:
                print(f"  [Images] Pass1 skip xref={xref} p{page_num + 1}: {e}")

    # ── Pass 2: vector / mixed figure pages ───────────────────────────────────
    scale = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)

    for page_num, page in enumerate(doc):
        # Already captured via Pass 1 with a known Figure caption — skip
        if page_num in raster_caption_pages:
            continue

        drawings  = page.get_drawings()
        page_text = page_texts[page_num]

        if len(drawings) < MIN_DRAW_PATHS:
            continue

        caption = _caption_on_page(page_text)

        # Accept if: has caption (confirmed figure) OR very sparse text (diagram-only page)
        if not caption and len(page_text) >= SPARSE_CHARS:
            continue

        try:
            pix = page.get_pixmap(matrix=scale, alpha=False)
            if pix.width < MIN_DIM or pix.height < MIN_DIM:
                continue

            image_id       = f"{doc_id}_p{page_num + 1}_render"
            image_filename = f"{image_id}.png"
            image_path     = IMAGES_DIR / image_filename
            pix.save(str(image_path))

            # If caption wasn't on this page, check the next page
            if not caption and page_num + 1 < len(page_texts):
                caption = _caption_on_page(page_texts[page_num + 1])

            images.append(_meta(image_id, image_filename, image_path,
                                page_num + 1, doc_id, source, caption, page_text))

        except Exception as e:
            print(f"  [Images] Pass2 skip p{page_num + 1}: {e}")

    doc.close()

    n_raster   = sum(1 for img in images if "_x" in img["image_id"])
    n_rendered = sum(1 for img in images if "_render" in img["image_id"])
    print(f"  [Images] {len(images)} image(s) from {source} "
          f"({n_raster} raster, {n_rendered} rendered)")
    return images


# ── Helpers ───────────────────────────────────────────────────────────────────

def _caption_on_page(page_text: str) -> str:
    """
    Return the first 'Figure N: description' caption whose label starts at
    the beginning of a line (possibly preceded by whitespace).

    This correctly rejects inline prose like "… as shown in Figure 1 …"
    while matching real captions like:
        Figure 1: The Transformer – model architecture.
        Fig. 2. Scaled Dot-Product Attention.
    """
    m = re.search(
        r"(?m)^\s*((?:Figure|Fig\.?)\s*\d+[:\.]?\s*.{5,200})",
        page_text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _meta(image_id: str, image_filename: str, image_path: Path,
          page: int, doc_id: str, source: str,
          caption: str, page_text: str) -> dict:
    return {
        "image_id":       image_id,
        "image_path":     str(image_path),
        "image_filename": image_filename,
        "page":           page,
        "doc_id":         doc_id,
        "source":         source,
        "caption":        caption,
        "context":        page_text[:500],
    }
