#!/usr/bin/env python3
"""
Recipe and Ingredient Extractor

Reads all parquet files (cooking book pages), uses a local LLM to extract
recipes and ingredients, then writes two output parquet files:
  1. recipes.parquet       – one row per recipe, with source book/page refs
  2. ingredients.parquet   – deduplicated index of all unique ingredients

Optimized pipeline:
  Pass 1: Keyword filter on text to identify recipe candidate pages
  Pass 2: LLM extracts structured recipe data from candidates only
  Pass 3: LLM normalizes ingredient lines into a unified ingredient index
"""

import os, sys, json, time, argparse, threading, signal
from pathlib import Path
from typing import Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI

BATCH_SIZE = int(os.environ.get("RECIPE_BATCH_SIZE", "5"))
FLUSH_INTERVAL = int(os.environ.get("RECIPE_FLUSH_INTERVAL", "5"))
MAX_CONCURRENT = int(os.environ.get("RECIPE_MAX_CONCURRENT", "8"))
CHECKPOINT_FILE = ".extract_recipes_checkpoint.json"

# ── Configuration ────────────────────────────────────────────────────────────
# Endpoints are env-overridable (the GUI sets these per run). Leave a base
# empty to disable that server.
LLM_BASE_1 = os.environ.get("LLM_BASE_1", "http://192.168.0.53:8040/v1")
LLM_BASE_2 = os.environ.get("LLM_BASE_2", "http://192.168.0.53:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "squig")
client_1 = OpenAI(base_url=LLM_BASE_1, api_key="not-needed") if LLM_BASE_1 else None
client_2 = OpenAI(base_url=LLM_BASE_2, api_key="not-needed") if LLM_BASE_2 else None

# Keywords that strongly suggest a page is a recipe (checked in text + summary)
RECIPE_TEXT_KEYWORDS = [
    "makes", "yield", "loaf", "bake", "baking", "baked", "oven",
    "preheat", "mix", "stir", "knead", "fold", "whisk", "beat",
    "simmer", "boil", "saute", "sauté", "fry", "grill", "roast",
    "broil", "steam", "poach", "blend", "chop", "dice", "mince",
    "cups", "teaspoon", "tablespoon", "tbsp", "tsp", "cups",
    "ounce", "pound", "gram", "ml", "liters", "quart", "pint",
    "large egg", "medium egg", "cloves", "sprig",
]

# Keywords that suggest it's NOT a recipe
NON_RECIPE_KEYWORDS = [
    "table of contents", "contents", "index", "copyright",
    "acknowledgments", "dedication", "preface", "foreword",
    "introduction", "introduction to", "about the author",
    "about the photographer", "photography by",
]

# ── LLM Prompts ──────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """\
Extract all recipes from the following cookbook page(s). Pages are separated by "--- PAGE BREAK ---" markers.

CRITICAL: Recipes often span multiple pages. A page break can occur MID-SENTENCE. When you see a sentence ending on one page that clearly continues on the next page (e.g., "Place the baking sheet on a rack so" / "that air can circulate"), you MUST merge them into the instructions. Do NOT stop instructions at a page break.

Look for:
- Recipe titles (often centered, capitalized, or bold in the original)
- Ingredient lists that start on one page and continue on the next
- Instructions that start on one page and continue on the next, sometimes mid-word or mid-sentence
- Yield lines (Yield:, Makes:, Serves:) may appear at the end of the last page

For each recipe provide:
- recipe_title: dish name
- description: intro/blurb text (empty string if none)
- ingredients: list of ingredient lines exactly as written (merged across pages if the recipe spans multiple)
- instructions: full step-by-step text COMPLETELY MERGED across page breaks. If a sentence is split across pages, combine it into one continuous text.
- yields: "Makes X" or "Yield: X" line if present
- page_nums: list of page numbers where this recipe appears
- food_type: primary category (e.g., "cookie", "pasta", "salad", "soup", "cake", "bread", "seafood", "chicken", "dessert", "appetizer", "sandwich", "stew")
- cuisine: cuisine style if identifiable (e.g., "Italian", "Chinese", "Mexican", "Japanese", "French", "American", "Indian", "Thai", "Southern", "Mediterranean"); empty string if not identifiable
- meal_type: which meal this is for (e.g., "breakfast", "lunch", "dinner", "dessert", "snack", "brunch", "appetizer", "brunch"); empty string if not clear

Return JSON only (no markdown, no explanation):
{{
  "recipes": [
    {{
      "recipe_title": "...",
      "description": "...",
      "ingredients": ["2 cups flour", "1 tsp salt"],
      "instructions": "...",
      "yields": "Makes 1 loaf",
      "page_nums": [12, 13],
      "food_type": "cookie",
      "cuisine": "American",
      "meal_type": "dessert"
    }}
  ]
}}
"""

INGREDIENT_PARSE_PROMPT = """\
Parse these ingredient lines into normalized ingredient names.
For each line: extract the base food item, lowercased, without quantity/measurement.
Split compound lines (joined by "and" or commas if they're separate ingredients).
Drop preparation notes (chopped, drained, etc.), but keep distinguishing adjectives (red bell pepper).

Input lines (JSON array):
{lines}

Return JSON only:
{{
  "ingredients": [
    {{"name": "flour", "original": "4 cups bread flour"}},
    ...
  ]
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def llm_chat(client_obj, messages: list[dict], max_tokens: int = 4096) -> str:
    resp = client_obj.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def is_recipe_page(text: str, summary: str = "") -> bool:
    """Lightweight keyword-based filter. Returns True if page is likely a recipe."""
    combined = (text + " " + summary).lower()
    if any(kw in combined for kw in NON_RECIPE_KEYWORDS):
        return False
    hits = sum(1 for kw in RECIPE_TEXT_KEYWORDS if kw in combined)
    return hits >= 2


def extract_recipes_from_page(text: str) -> list[dict]:
    """Call LLM to extract structured recipe(s) from page text."""
    out = llm.call([
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": text[:8000]},
    ])
    data = parse_json(out)
    return data.get("recipes", [])


def normalize_ingredient_lines(lines: list[str]) -> list[dict]:
    """Call LLM to parse ingredient lines into normalized names."""
    lines_json = json.dumps(lines)
    prompt = INGREDIENT_PARSE_PROMPT.format(lines=lines_json)
    out = llm.call([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Parse these ingredients."},
    ])
    return parse_json(out).get("ingredients", [])


# ── Thread-safe LLM queue for parallel requests ─────────────────────────────

class LLMQueue:
    """Round-robin load balancer across the configured LLM servers with a
    concurrency limit."""

    def __init__(self, clients: list, max_concurrent: int = 8):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._call_count = 0
        self._lock = threading.Lock()
        self._counter = 0
        self._clients = [c for c in clients if c is not None]

    def call(self, messages: list[dict], max_tokens: int = 4096) -> str:
        if not self._clients:
            raise RuntimeError("No LLM endpoint configured (set LLM_BASE_1 or LLM_BASE_2)")
        with self._semaphore:
            # Round-robin across all configured API servers
            idx = self._counter % len(self._clients)
            client_obj = self._clients[idx]
            with self._lock:
                self._counter += 1
            result = llm_chat(client_obj, messages, max_tokens)
            with self._lock:
                self._call_count += 1
        return result

    @property
    def calls(self) -> int:
        with self._lock:
            return self._call_count


llm = LLMQueue(clients=[client_1, client_2], max_concurrent=MAX_CONCURRENT)
_active = [b for b, c in ((LLM_BASE_1, client_1), (LLM_BASE_2, client_2)) if c]
print(f"[recipes] LLM endpoint(s): {', '.join(_active) or 'NONE'}  "
      f"model={LLM_MODEL}  max_concurrent={MAX_CONCURRENT}  "
      f"batch={BATCH_SIZE}  flush={FLUSH_INTERVAL}")
if not _active:
    print("[recipes] WARNING: no LLM endpoints active — extraction will fail.")


# ── Context for mutable state shared across functions ─────────────────────────

class ProcessContext:
    """Holds mutable accumulators so process_batch can update them."""

    def __init__(self):
        self.recipe_rows: List[dict] = []
        self.ingredient_rows: List[dict] = []
        self.ingredient_set: set = set()
        self.pages_scanned: int = 0
        self.pages_filtered: int = 0
        self.recipes_found: int = 0
        self.llm_errors: int = 0
        self.flush_counter: int = 0


def save_checkpoint(output_path: Path, ctx: ProcessContext, last_file: str,
                    processed_files: List[str]) -> None:
    """Persist processing state so we can resume after interruption."""
    cp = {
        "last_processed_file": last_file,
        "processed_files": processed_files,
        "pages_scanned": ctx.pages_scanned,
        "pages_filtered": ctx.pages_filtered,
        "recipes_found": ctx.recipes_found,
        "llm_errors": ctx.llm_errors,
        "flush_counter": ctx.flush_counter,
        "ingredient_rows": ctx.ingredient_rows,
        "ingredient_set": sorted(ctx.ingredient_set),
    }
    cp_path = output_path / CHECKPOINT_FILE
    with open(cp_path, "w") as f:
        json.dump(cp, f, indent=2)


def load_checkpoint(output_path: Path) -> Optional[dict]:
    """Load previously saved checkpoint, if any."""
    cp_path = output_path / CHECKPOINT_FILE
    if not cp_path.exists():
        return None
    try:
        with open(cp_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_checkpoint_into_ctx(cp: dict, ctx: ProcessContext) -> None:
    """Restore ProcessContext state from a checkpoint."""
    ctx.pages_scanned = cp["pages_scanned"]
    ctx.pages_filtered = cp["pages_filtered"]
    ctx.recipes_found = cp["recipes_found"]
    ctx.llm_errors = cp["llm_errors"]
    ctx.flush_counter = cp["flush_counter"]
    ctx.ingredient_rows = cp["ingredient_rows"]
    ctx.ingredient_set = set(cp["ingredient_set"])


def flush_writes(ctx: ProcessContext, output_path: Path) -> None:
    """Incrementally write accumulated rows to parquet files."""
    recipes_path = output_path / "recipes.parquet"
    ingredients_path = output_path / "ingredients.parquet"

    if ctx.recipe_rows:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(ctx.recipe_rows)), recipes_path)

    if ctx.ingredient_rows:
        df_ing = pd.DataFrame(ctx.ingredient_rows).sort_values("ingredient").reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(df_ing), ingredients_path)

    ctx.flush_counter += 1
    print(f"\n  [flush #{ctx.flush_counter}] {len(ctx.recipe_rows)} recipes, {len(ctx.ingredient_rows)} ingredients written so far")


def process_batch(pages: List[dict], ctx: ProcessContext) -> None:
    """Process a batch of consecutive candidate pages from the same book in a single LLM call."""
    batch_text = "\n\n--- PAGE BREAK (page {} of {}) ---\n\n".format(
        ", ".join(str(p["page_num"]) for p in pages), len(pages)
    ).join(p["text"] for p in pages)
    max_t = min(8000 * len(pages), 128000)

    try:
        out = llm.call([
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": batch_text},
        ], max_tokens=max_t)
        recipes = parse_json(out).get("recipes", [])
    except Exception:
        ctx.llm_errors += 1
        return

    if not recipes:
        ctx.pages_filtered += len(pages)
        return

    for recipe in recipes:
        title = recipe.get("recipe_title", "Untitled")
        ctx.recipes_found += 1

        recipe_page_nums = recipe.get("page_nums", [])

        # Find the first page in the batch that matches the recipe's page range
        if recipe_page_nums:
            src = next((p for p in pages if p["page_num"] in recipe_page_nums), pages[0])
        else:
            src = pages[0]

        # Collect all page nums as a comma-separated string
        page_nums_str = ", ".join(str(p["page_num"]) for p in pages)

        ctx.recipe_rows.append({
            "source_book": src["book_title"],
            "source_file": src["source_file"],
            "source_page": page_nums_str,
            "recipe_title": title,
            "recipe_description": recipe.get("description", ""),
            "recipe_ingredient_lines": recipe.get("ingredients", []),
            "recipe_instructions": recipe.get("instructions", ""),
            "recipe_yields": recipe.get("yields", ""),
            "recipe_food_type": recipe.get("food_type", ""),
            "recipe_cuisine": recipe.get("cuisine", ""),
            "recipe_meal_type": recipe.get("meal_type", ""),
            "book_title_orig": src["row"].get("book_title", ""),
            "page_num": page_nums_str,
            "text": src["text"],
            "summary": src["summary"],
            "page_width": src["row"].get("page_width", None),
            "page_height": src["row"].get("page_height", None),
        })

        ing_lines = recipe.get("ingredients", [])
        if ing_lines:
            try:
                parsed = normalize_ingredient_lines(ing_lines)
            except Exception:
                ctx.llm_errors += 1
                parsed = []

            for ing in parsed:
                name = ing.get("name", "").strip().lower()
                if name and name not in ctx.ingredient_set:
                    ctx.ingredient_set.add(name)
                    ctx.ingredient_rows.append({
                        "ingredient": name,
                        "original_line": ing.get("original", ""),
                        "source_book": src["book_title"],
                        "source_file": src["source_file"],
                        "source_page": page_nums_str,
                        "recipe_title": title,
                    })


def process_directory(input_dir: str, output_dir: str,
                      start_file: Optional[str] = None,
                      max_files: Optional[int] = None,
                      resume: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_path.glob("*.parquet"))
    if max_files:
        parquet_files = parquet_files[:max_files]

    # Determine which files to skip based on checkpoint or start_file
    processed_files = []
    ctx = ProcessContext()

    if resume:
        cp = load_checkpoint(output_path)
        if cp:
            load_checkpoint_into_ctx(cp, ctx)
            processed_files = cp["processed_files"]
            # Load existing recipes from parquet if it exists
            recipes_path = output_path / "recipes.parquet"
            if recipes_path.exists():
                ctx.recipe_rows = pd.read_parquet(recipes_path).to_dict(orient="records")
            print(f"Resuming from checkpoint: {len(processed_files)} files already processed, "
                  f"{ctx.recipes_found} recipes, {len(ctx.ingredient_rows)} ingredients")
        else:
            print("No checkpoint found, starting fresh.")
    elif start_file:
        found = False
        filtered = []
        for f in parquet_files:
            if f.name == start_file:
                found = True
            if found:
                filtered.append(f)
        if found:
            parquet_files = filtered
        else:
            print(f"Warning: start_file '{start_file}' not found, processing all.")

    # Remove already-processed files from the list
    if processed_files:
        file_set = set(processed_files)
        original_count = len(parquet_files)
        parquet_files = [f for f in parquet_files if f.name not in file_set]
        skipped = original_count - len(parquet_files)
        if skipped:
            print(f"Skipping {skipped} already-processed file(s)")

    page_groups = []  # List of groups; each group is consecutive recipe pages from same book
    current_group = []
    current_book = None
    current_file = None
    pending_groups = []  # Accumulate groups to batch multiple into one LLM call
    total = len(parquet_files)
    print(f"\nProcessing {total} remaining parquet file(s)\n")

    t0 = time.time()

    # Register signal handler for graceful shutdown
    def _shutdown(signum, frame):
        print(f"\n\nInterrupted! Saving checkpoint...")
        last_file = current_file if current_file else (parquet_files[fi].name if fi < len(parquet_files) else "")
        # Rebuild processed_files list from original order
        completed_files = [parquet_files[i].name for i in range(fi)]
        if current_file and current_file in [p.name for p in parquet_files]:
            completed_files.append(current_file)
        current_processed = list(set(processed_files + completed_files))
        all_names_full = [f.name for f in sorted(input_path.glob("*.parquet"))]
        current_processed_ordered = [n for n in all_names_full if n in current_processed]
        save_checkpoint(output_path, ctx, last_file, current_processed_ordered)
        flush_writes(ctx, output_path)
        print(f"Checkpoint saved. To resume, run with --resume\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for fi, pf in enumerate(parquet_files):
        book_start = time.time()
        book_title = pf.stem
        sys.stdout.write(f"\r[{fi+1:>4}/{total}] {pf.name}             ")
        sys.stdout.flush()

        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            ctx.llm_errors += 1
            # Flush any pending group before moving to next file
            if current_group:
                process_batch(current_group, ctx)
                page_groups.append(current_group)
                current_group = []
                current_book = None
            continue

        if "book_title" in df.columns and df["book_title"].iloc[0]:
            book_title = df["book_title"].iloc[0]

        prev_page_num = None
        prev_was_recipe = False
        non_recipe_streak = 0

        for _, row in df.iterrows():
            text = row.get("text", "")
            if not text or (isinstance(text, str) and len(text.strip()) == 0):
                continue

            ctx.pages_scanned += 1
            page_num = row.get("page_num", "?")
            summary = row.get("summary", "")

            is_recipe = is_recipe_page(text, summary)

            # Check page continuity
            is_consecutive = (
                prev_page_num is not None
                and isinstance(page_num, int)
                and isinstance(prev_page_num, int)
                and page_num == prev_page_num + 1
                and book_title == current_book
                and pf.name == current_file
            )

            if not is_recipe:
                ctx.pages_filtered += 1

                # If this page immediately follows a recipe page (consecutive),
                # include it as a likely continuation page
                if is_consecutive and prev_was_recipe and non_recipe_streak == 0:
                    non_recipe_streak += 1
                    is_recipe = True  # Treat it as a candidate continuation
                elif non_recipe_streak > 0 and non_recipe_streak < 3:
                    # Allow up to 2 non-recipe pages in a row as continuation
                    if is_consecutive:
                        non_recipe_streak += 1
                        is_recipe = True
                    else:
                        # Gap or non-consecutive: flush group
                        non_recipe_streak = 0
                        if current_group:
                            process_batch(current_group, ctx)
                            page_groups.append(current_group)
                            current_group = []
                            current_book = None
                        prev_was_recipe = False
                        continue
                else:
                    # Too many non-recipe pages; flush group and reset
                    if current_group:
                        process_batch(current_group, ctx)
                        page_groups.append(current_group)
                        current_group = []
                        current_book = None
                    non_recipe_streak = 0
                    prev_was_recipe = False
                    continue

            if is_recipe:
                # Start a new group if book changed or page gap detected
                if book_title != current_book or current_file != pf.name:
                    if current_group:
                        process_batch(current_group, ctx)
                        page_groups.append(current_group)
                        current_group = []
                    current_book = book_title
                    current_file = pf.name
                    non_recipe_streak = 0

                elif prev_page_num is not None and isinstance(page_num, int) and isinstance(prev_page_num, int):
                    if page_num - prev_page_num > 1 and not current_group:
                        # Gap and no active group; just start fresh
                        non_recipe_streak = 0
                    elif page_num - prev_page_num > 1 and current_group:
                        # Gap detected; flush current group and start new
                        process_batch(current_group, ctx)
                        page_groups.append(current_group)
                        current_group = []
                        non_recipe_streak = 0

                current_group.append({
                    "text": text[:8000],
                    "page_num": page_num,
                    "summary": summary,
                    "book_title": book_title,
                    "source_file": pf.name,
                    "row": row,
                })
                prev_was_recipe = True

            prev_page_num = page_num

            # Flush if group gets large enough
            if len(current_group) >= BATCH_SIZE:
                process_batch(current_group, ctx)
                page_groups.append(current_group)
                current_group = []

                # Periodic incremental flush
                if ctx.recipes_found >= FLUSH_INTERVAL and ctx.recipes_found % FLUSH_INTERVAL < BATCH_SIZE:
                    flush_writes(ctx, output_path)
                    save_checkpoint(output_path, ctx, pf.name,
                                    list(set(processed_files + [parquet_files[i].name for i in range(fi+1)])))
                    sys.stdout.write(f"\r[{fi+1:>4}/{total}] {pf.name}             ")
                    sys.stdout.flush()

        # End of file: flush any remaining group
        if current_group:
            process_batch(current_group, ctx)
            page_groups.append(current_group)
            current_group = []
            current_book = None
            current_file = None

        # Periodic flush after each file
        if ctx.recipes_found >= FLUSH_INTERVAL and ctx.recipes_found % FLUSH_INTERVAL < BATCH_SIZE * 2:
            flush_writes(ctx, output_path)

        # Save checkpoint after completing each file
        save_checkpoint(output_path, ctx, pf.name,
                        list(set(processed_files + [parquet_files[i].name for i in range(fi+1)])))

    # Process any remaining group after all files are done (shouldn't happen with the new logic)
    if current_group:
        process_batch(current_group, ctx)

    # Final flush
    flush_writes(ctx, output_path)

    # Remove checkpoint on successful completion
    cp_path = output_path / CHECKPOINT_FILE
    if cp_path.exists():
        cp_path.unlink()

    elapsed = time.time() - t0

    print(f"\n\n{'='*60}")
    print(f"  Pages scanned       : {ctx.pages_scanned}")
    print(f"  Pages filtered out  : {ctx.pages_filtered}")
    print(f"  LLM API calls       : {llm.calls}")
    print(f"  Recipes extracted   : {ctx.recipes_found}")
    print(f"  Unique ingredients  : {len(ctx.ingredient_rows)}")
    print(f"  LLM errors          : {ctx.llm_errors}")
    print(f"  Elapsed             : {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract recipes and ingredients from cookbook parquet files.")
    parser.add_argument("input_dir", help="Directory with cookbook parquet files")
    parser.add_argument("output_dir", help="Directory for output parquet files", default=".")
    parser.add_argument("--start-file", help="Start processing from this filename (skip earlier files)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit to N files")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (saves progress after each file)")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.start_file, args.max_files, args.resume)
