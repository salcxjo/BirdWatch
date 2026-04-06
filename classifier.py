# classifier.py — BirdWatch
# TFLite bird classifier with top-3 debug logging
# No external API dependencies

import numpy as np
import csv
import json
import os
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- Paths ---
MODEL_DIR = os.path.expanduser("~/BirdWatch/model")
CUSTOM_MODEL  = os.path.join(MODEL_DIR, "birdwatch_custom.tflite")
CUSTOM_LABELS = os.path.join(MODEL_DIR, "birdwatch_labels.json")
DEFAULT_MODEL  = os.path.join(MODEL_DIR, "birds_V1.tflite")
DEFAULT_LABELS = os.path.join(MODEL_DIR, "aiy_birds_V1_labelmap.csv")

# --- Tunable ---
CONFIDENCE_THRESHOLD = 0.15   # raise to 0.5+ once real birds appear


# --- Label loaders ---
def load_csv_labels(path):
    labels = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['id'])] = row['name']
    return labels

def load_json_labels(path):
    with open(path) as f:
        raw = json.load(f)
    # Support both {id: name} and {name: id} formats
    first_key = next(iter(raw))
    if isinstance(first_key, str) and not first_key.isdigit():
        return {v: k for k, v in raw.items()}
    return {int(k): v for k, v in raw.items()}


class BirdClassifier:
    def __init__(self):
        # Prefer custom fine-tuned model if available
        if os.path.exists(CUSTOM_MODEL) and os.path.exists(CUSTOM_LABELS):
            model_path = CUSTOM_MODEL
            self.labels = load_json_labels(CUSTOM_LABELS)
            print("Using custom fine-tuned model")
        else:
            model_path = DEFAULT_MODEL
            self.labels = load_csv_labels(DEFAULT_LABELS)
            print("Using default AIY birds model")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']
        self.input_size = input_shape[1]  # height (== width, square input)
        self.input_dtype = self.input_details[0]['dtype']
        print(f"Model loaded. Input: {self.input_size}x{self.input_size} {self.input_dtype.__name__}")

    def preprocess(self, frame_bgr):
        """Resize and convert to the dtype the model expects."""
        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB
        img = img.resize((self.input_size, self.input_size), Image.LANCZOS)

        if self.input_dtype == np.uint8:
            arr = np.array(img, dtype=np.uint8)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0

        return np.expand_dims(arr, axis=0)

    def classify(self, frame_bgr):
        """
        Run inference on a BGR frame.
        Returns (species_name, confidence, source)
        source is 'tflite' or 'unknown'
        """
        inp = self.preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Normalize to probabilities if raw logits
        output = output.astype(np.float32)
        if output.max() > 1.0:
            exp = np.exp(output - output.max())
            output = exp / exp.sum()

        # Suppress background class (index 0)
        output[0] = 0.0

        # Top-3 debug output
        top3 = np.argsort(output)[-3:][::-1]
        print("  Top 3:")
        for idx in top3:
            print(f"    [{idx}] {self.labels.get(idx, 'Unknown'):30s} {output[idx]:.1%}")

        top_idx = int(top3[0])
        confidence = float(output[top_idx])
        species = self.labels.get(top_idx, "Unknown")

        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"  → {species} ({confidence:.1%}) via tflite")
            return species, confidence, "tflite"

        print(f"  → Below threshold ({confidence:.1%}), skipping")
        return "Unknown", confidence, "unknown"
