#!/usr/bin/env python3
# checker.py — BirdWatch
# Re-runs inference on all saved detections, averages N passes,
# and reports where the model now disagrees with the original label.
#
# Usage:
#   python checker.py               # review all detections
#   python checker.py --passes 5    # average over 5 inference passes
#   python checker.py --update      # write corrected species back to DB

import argparse
import sqlite3
import os
import cv2
import numpy as np
from collections import defaultdict
from classifier import BirdClassifier

DB_PATH  = os.path.expanduser("~/BirdWatch/data/birdwatch.db")
PASSES   = 3   # default inference passes per image

def run_passes(classifier, frame, n):
    """
    Run inference n times on the same frame and average the raw output scores.
    Slight random crops add variance so averaging is meaningful.
    Returns (species, confidence, all_top1_names)
    """
    h, w = frame.shape[:2]
    scores_sum = None

    for i in range(n):
        # Slight random crop (±5%) to add variance
        if i > 0:
            dy = int(h * 0.05 * (np.random.rand() - 0.5))
            dx = int(w * 0.05 * (np.random.rand() - 0.5))
            y1 = max(0, dy); y2 = min(h, h + dy)
            x1 = max(0, dx); x2 = min(w, w + dx)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame

        inp = classifier.preprocess(crop)
        classifier.interpreter.set_tensor(
            classifier.input_details[0]['index'], inp)
        classifier.interpreter.invoke()
        output = classifier.interpreter.get_tensor(
            classifier.output_details[0]['index'])[0].astype(np.float32)

        # Normalize
        if output.max() > 1.0:
            exp = np.exp(output - output.max())
            output = exp / exp.sum()

        output[0] = 0.0  # suppress background

        if scores_sum is None:
            scores_sum = output.copy()
        else:
            scores_sum += output

    avg = scores_sum / n
    top_idx = int(np.argmax(avg))
    confidence = float(avg[top_idx])
    species = classifier.labels.get(top_idx, "Unknown")

    # Collect top-3
    top3 = [(classifier.labels.get(i, "?"), float(avg[i]))
            for i in np.argsort(avg)[-3:][::-1]]

    return species, confidence, top3


def main():
    parser = argparse.ArgumentParser(description="Re-run inference on all detections")
    parser.add_argument("--passes", type=int, default=PASSES,
                        help="Number of inference passes to average (default 3)")
    parser.add_argument("--update", action="store_true",
                        help="Write corrected species back to DB")
    parser.add_argument("--only-unknown", action="store_true",
                        help="Only re-check Unknown/background detections")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Only show results above this confidence")
    args = parser.parse_args()

    classifier = BirdClassifier()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM detections ORDER BY timestamp DESC"
    if args.only_unknown:
        query = """SELECT * FROM detections
                   WHERE species IN ('Unknown','background')
                   ORDER BY timestamp DESC"""

    rows = conn.execute(query).fetchall()
    print(f"\nChecking {len(rows)} detections with {args.passes} passes each...\n")

    changed = 0
    errors  = 0
    species_corrections = defaultdict(list)

    for row in rows:
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"  MISSING: {img_path}")
            errors += 1
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  UNREADABLE: {img_path}")
            errors += 1
            continue

        new_species, confidence, top3 = run_passes(classifier, frame, args.passes)

        original = row['species']
        ts = row['timestamp'][:16]
        fname = os.path.basename(img_path)

        if confidence < args.min_confidence:
            continue

        status = "✓" if new_species == original else "≠"
        if new_species != original:
            changed += 1
            species_corrections[original].append(new_species)

        print(f"[{status}] {ts}  {fname}")
        print(f"      Original : {original}")
        print(f"      New      : {new_species} ({confidence:.1%})")
        print(f"      Top 3    : {', '.join(f'{n}({c:.0%})' for n,c in top3)}")
        print()

        if args.update and new_species != original:
            conn.execute(
                """UPDATE detections
                   SET species=?, confidence=?, classifier_source='tflite_recheck'
                   WHERE id=?""",
                (new_species, confidence, row['id'])
            )

    if args.update:
        conn.commit()
        print(f"DB updated.")

    conn.close()

    # Summary
    print("=" * 60)
    print(f"Total checked : {len(rows)}")
    print(f"Changed       : {changed}")
    print(f"Errors        : {errors}")
    if species_corrections:
        print("\nCorrection breakdown:")
        for original, new_list in sorted(species_corrections.items()):
            from collections import Counter
            for new, count in Counter(new_list).most_common():
                print(f"  {original:35s} → {new} ({count}x)")


if __name__ == "__main__":
    main()
