#!/usr/bin/env python3
# organize_dataset.py — BirdWatch
# Copies high-confidence detections from the DB into species folders
# for use as training data in the fine-tuning pipeline.
#
# Usage:
#   python organize_dataset.py
#   python organize_dataset.py --min-confidence 0.90

import sqlite3
import os
import shutil
import argparse

DB_PATH    = os.path.expanduser("~/BirdWatch/data/birdwatch.db")
OUTPUT_DIR = os.path.expanduser("~/BirdWatch/dataset/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-confidence", type=float, default=0.75,
                        help="Minimum confidence to include (default 0.75)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT image_path, species, confidence FROM detections
        WHERE species NOT IN ('Unknown', 'background')
        AND confidence >= ?
        ORDER BY species
    """, (args.min_confidence,)).fetchall()
    conn.close()

    counts = {}
    skipped = 0

    for img_path, species, conf in rows:
        if not os.path.exists(img_path):
            skipped += 1
            continue
        folder = species.replace(' ', '_')
        species_dir = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(species_dir, exist_ok=True)
        dst = os.path.join(species_dir, os.path.basename(img_path))
        if not os.path.exists(dst):
            shutil.copy2(img_path, dst)
        counts[species] = counts.get(species, 0) + 1

    print(f"\nDataset organized (min confidence: {args.min_confidence:.0%})")
    print(f"Skipped {skipped} missing files\n")
    for species, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * min(count, 40)
        note = " ✓" if count >= 30 else f" (need {30-count} more)"
        print(f"  {species:35s} {count:3d}  {bar}{note}")
    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
