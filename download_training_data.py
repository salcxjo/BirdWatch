#!/usr/bin/env python3
# download_training_data.py — BirdWatch
# Downloads labeled bird photos from GBIF for Edmonton-area species.
# No API key required. Run on your laptop or Pi.
#
# Usage:
#   python download_training_data.py                  # download all species
#   python download_training_data.py --species "Pica hudsonia"  # one species
#   python download_training_data.py --limit 80       # max images per species
#   python download_training_data.py --dry-run        # show counts only

import os
import time
import argparse
import requests
from pathlib import Path

# --- Edmonton-area species to download ---
EDMONTON_SPECIES = [
    ("Pica hudsonia",           "Black-billed_Magpie"),
    ("Corvus brachyrhynchos",   "American_Crow"),
    ("Poecile atricapillus",    "Black-capped_Chickadee"),
    ("Sitta carolinensis",      "White-breasted_Nuthatch"),
    ("Haemorhous mexicanus",    "House_Finch"),
    ("Spinus tristis",          "American_Goldfinch"),
    ("Passer domesticus",       "House_Sparrow"),
    ("Turdus migratorius",      "American_Robin"),
    ("Junco hyemalis",          "Dark-eyed_Junco"),
    ("Picoides pubescens",      "Downy_Woodpecker"),
    ("Dryobates villosus",      "Hairy_Woodpecker"),
    ("Colaptes auratus",        "Northern_Flicker"),
    ("Bombycilla cedrorum",     "Cedar_Waxwing"),
    ("Bombycilla garrulus",     "Bohemian_Waxwing"),
    ("Columba livia",           "Rock_Pigeon"),
    ("Streptopelia decaocto",   "Eurasian_Collared-Dove"),
    ("Sturnus vulgaris",        "European_Starling"),
    ("Branta canadensis",       "Canada_Goose"),
    ("Anas platyrhynchos",      "Mallard"),
    ("Falco sparverius",        "American_Kestrel"),
    ("Accipiter cooperii",      "Coopers_Hawk"),
    ("Buteo jamaicensis",       "Red-tailed_Hawk"),
    ("Spizella passerina",      "Chipping_Sparrow"),
    ("Melospiza melodia",       "Song_Sparrow"),
    ("Zonotrichia leucophrys",  "White-crowned_Sparrow"),
    ("Setophaga petechia",      "Yellow_Warbler"),
]

# Edmonton bounding box — broad enough to include surrounding area
# lat: 52.5 to 54.5, lon: -115 to -112
EDMONTON_LAT = "52.5,54.5"
EDMONTON_LON = "-115,-112"

GBIF_API = "https://api.gbif.org/v1/occurrence/search"
DATASET_DIR = os.path.expanduser("~/BirdWatch/dataset")

def get_gbif_observations(scientific_name, limit=100, offset=0):
    """Fetch observations with photos near Edmonton from GBIF."""
    params = {
        "scientificName": scientific_name,
        "decimalLatitude": EDMONTON_LAT,
        "decimalLongitude": EDMONTON_LON,
        "mediaType": "StillImage",
        "limit": min(limit, 300),
        "offset": offset,
        "hasCoordinate": "true",
    }
    try:
        r = requests.get(GBIF_API, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  GBIF API error: {e}")
        return {"results": [], "count": 0}

def extract_image_urls(observation):
    """Pull image URLs from a GBIF observation record."""
    urls = []
    for media in observation.get("media", []):
        url = media.get("identifier", "")
        # Accept jpg/jpeg only — skip tiff, video etc
        if url and any(url.lower().endswith(ext) for ext in ('.jpg', '.jpeg')):
            urls.append(url)
    return urls

def download_image(url, dest_path, session):
    """Download a single image. Returns True on success."""
    try:
        r = session.get(url, timeout=12, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        # Sanity check — reject tiny files (likely error pages)
        if os.path.getsize(dest_path) < 5000:
            os.remove(dest_path)
            return False
        return True
    except Exception:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_species(scientific_name, folder_name, limit, dry_run, session):
    out_dir = Path(DATASET_DIR) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = list(out_dir.glob("*.jpg"))
    already = len(existing)
    need = max(0, limit - already)

    print(f"\n{'='*55}")
    print(f"  {folder_name.replace('_',' ')} ({scientific_name})")
    print(f"  Already have: {already} | Target: {limit} | Need: {need}")

    if need == 0:
        print(f"  Already at target — skipping")
        return already, 0

    # Fetch observations in pages
    downloaded = 0
    offset = 0
    page_size = 100

    while downloaded < need:
        data = get_gbif_observations(scientific_name, limit=page_size, offset=offset)
        results = data.get("results", [])
        total_available = data.get("count", 0)

        if not results:
            print(f"  No more results (total available: {total_available})")
            break

        for obs in results:
            if downloaded >= need:
                break
            urls = extract_image_urls(obs)
            for url in urls:
                if downloaded >= need:
                    break
                # Derive filename from GBIF key + url hash
                gbif_key = obs.get("key", "unknown")
                ext = ".jpg"
                fname = f"gbif_{gbif_key}_{downloaded:04d}{ext}"
                fpath = out_dir / fname

                if fpath.exists():
                    continue

                if dry_run:
                    print(f"  [DRY RUN] Would download: {url[:60]}...")
                    downloaded += 1
                    continue

                success = download_image(url, fpath, session)
                if success:
                    downloaded += 1
                    if downloaded % 10 == 0:
                        print(f"  Downloaded {downloaded}/{need}...")
                else:
                    pass  # silently skip failed downloads

                time.sleep(0.15)  # be polite to GBIF servers

        offset += page_size
        if offset >= min(total_available, 900):  # GBIF caps at 300/page, 3 pages max
            break

    total = already + downloaded
    print(f"  Done: {downloaded} new, {total} total")
    return total, downloaded

def main():
    parser = argparse.ArgumentParser(
        description="Download labeled bird training images from GBIF"
    )
    parser.add_argument("--species", type=str, default=None,
                        help="Scientific name of one species to download")
    parser.add_argument("--limit", type=int, default=60,
                        help="Target images per species (default 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "BirdWatch/1.0 (bird monitoring project)"})

    if args.species:
        # Find matching entry
        match = next(
            ((sci, folder) for sci, folder in EDMONTON_SPECIES
             if sci.lower() == args.species.lower()),
            None
        )
        if not match:
            print(f"Species '{args.species}' not in Edmonton list.")
            print("Available:", ", ".join(s for s, _ in EDMONTON_SPECIES))
            return
        download_species(match[0], match[1], args.limit, args.dry_run, session)
    else:
        total_new = 0
        summary = []
        for sci, folder in EDMONTON_SPECIES:
            total, new = download_species(sci, folder, args.limit, args.dry_run, session)
            total_new += new
            summary.append((folder.replace('_',' '), total))

        print(f"\n{'='*55}")
        print(f"SUMMARY — {total_new} new images downloaded")
        print(f"{'='*55}")
        for name, count in summary:
            bar = '█' * min(count, 60)
            print(f"  {name:30s} {count:3d}  {bar}")

    print(f"\nDataset at: {DATASET_DIR}")
    print("Next step: run organize_dataset.py, then fine-tune on Colab")

if __name__ == "__main__":
    main()
