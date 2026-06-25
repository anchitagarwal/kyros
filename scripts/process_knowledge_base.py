"""
scripts/process_knowledge_base.py

Preprocesses raw Discord HTML exports (+ image attachments) into clean
markdown files the Kyros Planner can actually read.

Key design principles
─────────────────────
1. Filter-first: ICT keyword filter runs immediately after HTML parsing,
   before any image captioning. Only images attached to messages that pass
   the filter get captioned — not the full 7K+ image corpus.

2. Parallel captioning: Images are captioned with a ThreadPoolExecutor
   (default 20 workers), cutting runtime by ~15-18x vs sequential.

3. Checkpoint saves: Caption cache is written every 50 completions so a
   killed run loses at most ~50 captions worth of work.

4. --dry-run: Reports filter stats, image count, estimated cost/time
   without making any API calls or writing any files. Run this first.

5. --no-captions: Fast text-only pass with no API calls. Useful for
   validating HTML parsing and filter quality before spending money.

Usage
─────
Step 1 — dry run to see stats and estimated cost:
    uv run --env-file .env python scripts/process_knowledge_base.py \\
        --input ~/src/kyros/discord_ttt_dump/day_trade_alerts \\
        --images ~/src/kyros/discord_ttt_dump/day_trade_alerts/media/attachments \\
        --output workspace/knowledge_base \\
        --dry-run

Step 2 — fast text-only pass (no API calls, instant output):
    uv run --env-file .env python scripts/process_knowledge_base.py \\
        --input ~/src/kyros/discord_ttt_dump/day_trade_alerts \\
        --output workspace/knowledge_base \\
        --no-captions

Step 3 — full run with parallel image captioning:
    uv run --env-file .env python scripts/process_knowledge_base.py \\
        --input ~/src/kyros/discord_ttt_dump/day_trade_alerts \\
        --images ~/src/kyros/discord_ttt_dump/day_trade_alerts/media/attachments \\
        --output workspace/knowledge_base

Run once per channel (day_trade_alerts, education) — both append to the
same output dir and cache file.
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from bs4 import BeautifulSoup


# ── ICT keyword filter ────────────────────────────────────────────────────────

ICT_KEYWORDS = [
    "FVG", "fair value gap",
    "MSS", "market structure shift",
    "BOS", "break of structure",
    "ChoCH", "change of character",
    "swing high", "swing low",
    "HH", "HL", "LH", "LL",
    "IFVG", "implied fair value",
    "volume imbalance",
    "opening gap", "NWOG", "NDOG",
    "order block", " OB ", "OB.",
    "breaker block", "breaker",
    "mitigation block", "rejection block",
    "BSL", "SSL",
    "buy side", "sell side",
    "equal highs", "equal lows",
    "PDH", "PDL", "PWH", "PWL",
    "liquidity sweep", "liquidity void",
    "REL", "relative equal",
    "stop hunt",
    "premium", "discount",
    "equilibrium",
    "OTE", "optimal trade entry",
    "kill zone", "killzone",
    "asian range", "london open", "ny open",
    "silver bullet",
    "macro", "9:30", "10:00", "2:00",
    "PO3", "power of three",
    "accumulation", "manipulation", "distribution",
    "displacement", "displaced", "impulse",
    "SMT", "smart money technique", "divergence",
    "inducement", "IDM",
    "turtle soup",
    "ICT", "inner circle trader",
    "PDA", "premium discount array",
    "HTF", "LTF",
    "CISD", "change in state of delivery",
    "inversion", "inverted FVG",
    "1m MSS", "5m FVG", "15m FVG",
    "mitigation", "mitigated",
    "VI",     # volume imbalance shorthand used in alerts
    "RB",     # rejection block shorthand
    "IRL",    # internal range liquidity
    "ERL",    # external range liquidity
    "IOFED",  # ICT entry model
    "pHOD", "pLOD",  # prior high/low of day
]

_ICT_RE = re.compile(
    "|".join(r"\b" + re.escape(kw) + r"\b" for kw in ICT_KEYWORDS),
    re.IGNORECASE,
)

_NOISE_RE = re.compile(
    r"(apex|topstep|alpha futures|prop firm|eval|funded account|payout|"
    r"ninjatrader|tradingview platform|platform fee|profit split|"
    r"trailing drawdown|dividend|long.term invest)",
    re.IGNORECASE,
)

_CUSTOM_EMOJI_RE = re.compile(r":[a-zA-Z0-9_]+:")


def is_ict_relevant(text: str) -> bool:
    matches = _ICT_RE.findall(text)
    distinct = {m.upper() for m in matches}
    if len(distinct) < 3:
        return False
    noise_hits = len(_NOISE_RE.findall(text))
    if noise_hits > 0 and len(matches) <= noise_hits:
        return False
    return True


def ict_score(text: str) -> int:
    """Count of distinct ICT keywords — higher = more signal-dense."""
    return len({m.upper() for m in _ICT_RE.findall(text)})


def apply_token_budget(messages, max_tokens, enc):
    scored = sorted(enumerate(messages), key=lambda x: ict_score(x[1]["text"]), reverse=True)
    kept_indices = set()
    used = 0
    for original_idx, msg in scored:
        text_tokens = len(enc.encode(msg["text"]))
        cap_text    = " ".join(msg.get("captions", []))
        cap_tokens  = len(enc.encode(cap_text)) if cap_text else 0
        msg_tokens  = text_tokens + cap_tokens
        if used + msg_tokens > max_tokens:
            break
        kept_indices.add(original_idx)
        used += msg_tokens
    result = [m for i, m in enumerate(messages) if i in kept_indices]
    print(f"  token budget: kept {len(result):,} of {len(messages):,} messages ({used:,} tokens)")
    return result


# ── HTML parsing (no captioning) ──────────────────────────────────────────────

def parse_html_file(path: Path) -> list[dict]:
    """
    Extract messages from a Discord HTML export file.

    Deliberately does NOT caption images — it only records image filenames.
    Captioning happens after the ICT filter so we skip images attached to
    messages that would be dropped anyway.
    """
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    messages = []
    for msg in soup.select(".message"):
        text_el = msg.select_one(".message-text")
        if not text_el:
            continue

        # Expand emoji images to their alt text
        for img in text_el.select("img.emoji"):
            img.replace_with(img.get("alt", ""))

        for br in text_el.find_all("br"):
            br.replace_with("\n")

        text = text_el.get_text()
        text = _CUSTOM_EMOJI_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+",  " ",    text)
        text = text.strip()

        if not text or len(text) < 5:
            continue

        ts_el     = msg.select_one(".timestamp")
        timestamp = ts_el.get_text(strip=True) if ts_el else ""

        # Collect image filenames only — captioning deferred until after filter
        image_filenames = []
        for img_tag in msg.select("img.attachment-preview-img"):
            src = img_tag.get("src", "")
            fn  = src.split("/")[-1]
            if fn:
                image_filenames.append(fn)

        messages.append({
            "timestamp":       timestamp,
            "text":            text,
            "image_filenames": image_filenames,
            "captions":        [],  # populated later by attach_captions()
        })

    return messages


# ── Parallel image captioning ─────────────────────────────────────────────────

_CAPTION_PROMPT = """You are analyzing a trading chart image shared in an ICT (Inner Circle Trader) futures trading community.

Describe the following in 3-5 sentences:
1. Ticker, timeframe, and approximate date/time shown
2. Any ICT concepts visible: FVG (fair value gaps), order blocks, BSL/SSL (buy/sell side liquidity), BOS/ChoCH (break of structure / change of character), displacement, kill zones, NWOG/NDOG, swing highs/lows, premium/discount zones, OTE levels
3. Key price levels marked on the chart
4. What the chart appears to be illustrating or the trade setup being shown

Be specific about price levels and ICT terminology. If no ICT concepts are visible, describe what is shown."""


def _caption_one(image_path: Path, client: anthropic.Anthropic) -> tuple[str, str]:
    """
    Caption a single chart image. Returns (filename, caption_text).
    Retries up to 3x on rate limit errors with exponential backoff.
    """
    suffix = image_path.suffix.lower()
    media_type_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/png")
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }},
                        {"type": "text", "text": _CAPTION_PROMPT},
                    ],
                }],
            )
            return image_path.name, resp.content[0].text.strip()

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n  rate limit — waiting {wait}s before retry {attempt + 1}/3...")
            time.sleep(wait)

        except Exception as e:
            return image_path.name, f"[Caption failed: {e}]"

    return image_path.name, "[Caption failed: rate limit after 3 retries]"


def caption_images_parallel(
    image_paths: list[Path],
    client: anthropic.Anthropic,
    cache: dict,
    cache_path: Path,
    max_workers: int = 20,
    checkpoint_every: int = 50,
) -> dict:
    """
    Caption images in parallel, skipping already-cached ones.
    Saves cache checkpoints every `checkpoint_every` completions so
    a killed run loses minimal work.
    Returns the updated cache dict.
    """
    to_caption = [p for p in image_paths if p.name not in cache]
    already    = len(image_paths) - len(to_caption)

    if already:
        print(f"  {already:,} already cached, skipping")
    if not to_caption:
        print(f"  nothing new to caption")
        return cache

    print(f"  captioning {len(to_caption):,} images with {max_workers} parallel workers...")
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_caption_one, p, client): p
            for p in to_caption
        }
        for future in as_completed(futures):
            filename, caption = future.result()
            cache[filename] = caption
            completed += 1

            if completed % checkpoint_every == 0:
                cache_path.write_text(json.dumps(cache, indent=2))
                pct = completed / len(to_caption) * 100
                print(f"  [{completed:,}/{len(to_caption):,} — {pct:.0f}%] checkpoint saved")

    # Final save
    cache_path.write_text(json.dumps(cache, indent=2))
    print(f"  {len(to_caption):,} images captioned")
    return cache


def attach_captions(messages: list[dict], images_dir: Path, cache: dict) -> None:
    """
    Populate the 'captions' field on each message from the cache.
    Silently skips images not on disk and failed captions.
    """
    for msg in messages:
        captions = []
        for fn in msg.get("image_filenames", []):
            if not (images_dir / fn).exists():
                continue
            cap = cache.get(fn, "")
            if cap and not cap.startswith("[Caption failed"):
                captions.append(cap)
        msg["captions"] = captions


# ── Markdown rendering ────────────────────────────────────────────────────────

def messages_to_markdown(
    messages: list[dict],
    title: str,
    source_files: list[str],
    ict_only: bool,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Source: {', '.join(source_files)}",
        f"Messages: {len(messages)}",
        f"Filter: {'ICT-relevant only' if ict_only else 'all messages'}",
        "",
        "---",
        "",
    ]

    for msg in messages:
        if msg["timestamp"]:
            lines.append(f"**[{msg['timestamp']}]**")

        # Chart captions inline before message text so the Planner reads
        # the visual context before the verbal context
        for i, cap in enumerate(msg.get("captions", []), 1):
            label = "Chart" if len(msg["captions"]) == 1 else f"Chart {i}"
            lines.append(f"> **[{label}]** {cap}")

        lines.append(msg["text"])
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def process(
    input_dir:   Path,
    images_dir:  Path | None,
    output_dir:  Path,
    dry_run:     bool = False,
    no_captions: bool = False,
    max_workers: int  = 20,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    alert_files = sorted(input_dir.glob("day_trade_alerts*.html"))
    edu_files   = sorted(input_dir.glob("education*.html"))

    if not alert_files and not edu_files:
        print(f"No matching HTML files found in {input_dir}")
        sys.exit(1)

    # ── 1. Parse HTML (zero API calls) ────────────────────────────────────────

    def load_channel(files: list[Path], label: str) -> list[dict]:
        all_msgs = []
        for i, f in enumerate(files, 1):
            sys.stdout.write(f"\r  [{i}/{len(files)}] {f.name:<60}")
            sys.stdout.flush()
            all_msgs.extend(parse_html_file(f))
        print(f"\r  {len(all_msgs):,} messages parsed from {len(files)} file(s){' ' * 30}")
        return all_msgs

    print(f"\nParsing {len(alert_files)} alert file(s)...")
    alert_msgs = load_channel(alert_files, "alerts")

    print(f"Parsing {len(edu_files)} education file(s)...")
    edu_msgs = load_channel(edu_files, "education")

    # ── 2. ICT filter — BEFORE any captioning ─────────────────────────────────
    #
    # Keep a message if its text is ICT-relevant OR it has image attachments
    # (images-only messages are chart posts that likely show ICT setups even
    # when the accompanying text is minimal like "NQ FVG").

    alert_ict = [m for m in alert_msgs if is_ict_relevant(m["text"]) or m["image_filenames"]]
    edu_ict   = [m for m in edu_msgs   if is_ict_relevant(m["text"]) or m["image_filenames"]]

    print(f"\nICT filter results:")
    print(f"  alerts    : {len(alert_msgs):>6,} → {len(alert_ict):>6,} kept  "
          f"({len(alert_msgs) - len(alert_ict):,} dropped)")
    print(f"  education : {len(edu_msgs):>6,} → {len(edu_ict):>6,} kept  "
          f"({len(edu_msgs) - len(edu_ict):,} dropped)")

    # ── 3. Image captioning (filtered messages only) ───────────────────────────

    if images_dir and not images_dir.exists():
        print(f"\nWarning: --images dir not found: {images_dir}. Skipping captions.")
        images_dir = None

    if images_dir and not no_captions:
        cache_path = output_dir / "image_cache.json"
        cache: dict = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            print(f"\nLoaded {len(cache):,} cached captions from {cache_path.name}")

        # Collect unique image filenames from filtered messages only
        all_filenames: set[str] = set()
        for msg in alert_ict + edu_ict:
            all_filenames.update(msg["image_filenames"])

        image_paths = [images_dir / fn for fn in all_filenames if (images_dir / fn).exists()]
        missing     = len(all_filenames) - len(image_paths)
        uncached    = [p for p in image_paths if p.name not in cache]

        print(f"\nImages (filtered messages only — not the full corpus):")
        print(f"  referenced in kept messages : {len(all_filenames):,}")
        print(f"  found on disk               : {len(image_paths):,}")
        print(f"  already cached              : {len(image_paths) - len(uncached):,}")
        print(f"  need captioning             : {len(uncached):,}")
        if missing:
            print(f"  not found on disk           : {missing:,}  (referenced but absent)")

        if dry_run:
            cost_usd = len(uncached) * 0.003
            mins     = (len(uncached) / max_workers) * 2 / 60  # ~2s per image
            print(f"\n[DRY RUN] Would caption {len(uncached):,} images")
            print(f"  Estimated API cost : ~${cost_usd:.2f}")
            print(f"  Estimated runtime  : ~{mins:.0f} min  ({max_workers} workers)")
            print(f"\nRe-run without --dry-run to proceed.")
            return

        if uncached:
            print(f"\nCaptioning...")
            client = anthropic.Anthropic()
            cache  = caption_images_parallel(
                image_paths, client, cache, cache_path,
                max_workers=max_workers,
            )

        attach_captions(alert_ict, images_dir, cache)
        attach_captions(edu_ict,   images_dir, cache)

    elif dry_run:
        # dry-run without --images: just show what text output would look like
        print(f"\n[DRY RUN] No --images provided. Text-only output would be:")
        print(f"  alerts_ict.md    : {len(alert_ict):,} messages")
        print(f"  education_ict.md : {len(edu_ict):,} messages")
        return

    # ── 4. Write output files ──────────────────────────────────────────────────

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 145_000

    if alert_ict:
        print("\nApplying token budget to alerts...")
        alert_ict = apply_token_budget(alert_ict, MAX_TOKENS, enc)
    if edu_ict:
        print("Applying token budget to education...")
        edu_ict = apply_token_budget(edu_ict, MAX_TOKENS, enc)

    outputs = {}

    if alert_files:
        outputs["alerts_ict.md"] = messages_to_markdown(
            alert_ict, "Day Trade Alerts — ICT Concepts",
            [f.name for f in alert_files], ict_only=True,
        )
        outputs["alerts_all.md"] = messages_to_markdown(
            alert_msgs, "Day Trade Alerts — All Messages",
            [f.name for f in alert_files], ict_only=False,
        )

    if edu_files:
        outputs["education_ict.md"] = messages_to_markdown(
            edu_ict, "Education Channel — ICT Concepts",
            [f.name for f in edu_files], ict_only=True,
        )
        outputs["education_all.md"] = messages_to_markdown(
            edu_msgs, "Education Channel — All Messages",
            [f.name for f in edu_files], ict_only=False,
        )

    print()
    for filename, content in outputs.items():
        out_path = output_dir / filename
        out_path.write_text(content, encoding="utf-8")
        kb    = len(content.encode()) / 1024
        lines = content.count("\n")
        print(f"  wrote {out_path}  ({kb:.0f} KB, {lines:,} lines)")

    print("\nDone.")
    if no_captions:
        print("Re-run with --images <path> to add chart captions.")
    elif images_dir:
        print("Chart captions are inlined as blockquotes in the *_ict.md files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Discord HTML exports into Kyros knowledge base markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # dry run — see stats and cost estimate before spending anything
  python scripts/process_knowledge_base.py \\
      --input discord_ttt_dump/day_trade_alerts \\
      --images discord_ttt_dump/day_trade_alerts/media/attachments \\
      --dry-run

  # fast text-only pass (no API calls)
  python scripts/process_knowledge_base.py \\
      --input discord_ttt_dump/day_trade_alerts \\
      --no-captions

  # full run with parallel captioning
  python scripts/process_knowledge_base.py \\
      --input discord_ttt_dump/day_trade_alerts \\
      --images discord_ttt_dump/day_trade_alerts/media/attachments
        """,
    )
    parser.add_argument(
        "--input", default=".",
        help="Directory containing Discord HTML export files",
    )
    parser.add_argument(
        "--images", default=None,
        help="Directory containing image attachments (media/attachments/)",
    )
    parser.add_argument(
        "--output", default="workspace/knowledge_base",
        help="Output directory (default: workspace/knowledge_base)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report filter stats and cost estimate; do not caption or write files",
    )
    parser.add_argument(
        "--no-captions", action="store_true",
        help="Skip image captioning entirely (fast text-only pass)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=20,
        help="Parallel workers for image captioning (default: 20)",
    )
    args = parser.parse_args()

    process(
        input_dir   = Path(args.input),
        images_dir  = Path(args.images) if args.images else None,
        output_dir  = Path(args.output),
        dry_run     = args.dry_run,
        no_captions = args.no_captions,
        max_workers = args.max_workers,
    )


if __name__ == "__main__":
    main()