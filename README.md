# 🐦 BirdWatch

**Raspberry Pi 4 balcony bird detection, classification, and logging system.**

BirdWatch uses a Pi Camera, OpenCV motion detection, and a TFLite MobileNet classifier to automatically detect and identify bird species visiting a balcony — logging every visit to a SQLite database and serving a live web dashboard over a local network.

Built as part of an ongoing journey learning computer vision and edge ML deployment on embedded hardware.

---

## Demo

| Dashboard | Species Gallery | Visit Log |
|-----------|----------------|-----------|
| Live latest detection, hourly activity charts | Side-by-side your photo vs Wikipedia reference | Tile grid of all confirmed visits |

---

## Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 4 | Main compute — detection, classification, web server |
| Pi Camera v2 (imx219) | 1640×1232 MJPEG stream via `rpicam-vid` |

---

## Features

- **Motion detection** via OpenCV MOG2 background subtraction
- **Burst capture** — collects up to 8 frames per motion event, classifies all, saves the best crop
- **Bird classification** via Google AIY Birds V1 TFLite model (965 species)
- **Edmonton species filter** — suppresses non-local species to reduce false IDs
- **SQLite logging** — every detection stored with timestamp, species, confidence, source
- **Flask web UI** — dashboard, species gallery with Wikipedia reference images, visit log
- **Email alerts** — configurable per-species email notifications
- **Sunrise/sunset shutoff** — stops detection at night using `astral`
- **Frame cropping** — saves tight crop around largest motion contour, not full frame
- **Checker script** — re-runs inference on saved detections with multi-pass averaging
- **Training data downloader** — fetches labeled photos from GBIF for fine-tuning
- **Systemd services** — both detector and web UI start automatically on boot

---

## Project Structure

```
BirdWatch/
├── detector.py              # Main detection loop — camera, motion, burst, classify
├── classifier.py            # TFLite inference + Edmonton species filter
├── app.py                   # Flask web UI
├── alerts.py                # Email alert system
├── checker.py               # Re-run inference on existing detections
├── organize_dataset.py      # Organizes detections into training folders
├── download_training_data.py# Downloads labeled photos from GBIF
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── gallery.html
│   └── logs.html
├── static/
│   └── style.css
├── model/                   # Place .tflite and label files here
│   ├── birds_V1.tflite      # Download separately (see Setup)
│   └── aiy_birds_V1_labelmap.csv
├── data/                    # SQLite DB (auto-created)
├── detections/              # Saved detection images (auto-created)
└── logs/                    # Service logs (auto-created)
```

---

## Setup

### 1. System dependencies

```bash
sudo apt update
sudo apt install -y rpicam-apps python3-pip
```

### 2. Python environment

```bash
# Install pyenv + Python 3.11 (required — tflite-runtime not available for 3.12+)
curl https://pyenv.run | bash
source ~/.bashrc
pyenv install 3.11.9
cd ~/BirdWatch && pyenv local 3.11.9
python -m venv venv --system-site-packages
source venv/bin/activate

pip install opencv-python-headless numpy pillow flask paho-mqtt \
            requests astral python-dotenv tflite-runtime
```

> **TFLite wheel for aarch64 + Python 3.11:**
> ```bash
> pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp311-cp311-linux_aarch64.whl
> ```
> If unavailable, install full TensorFlow — `classifier.py` falls back to `tensorflow.lite` automatically.

### 3. Download the model

```bash
mkdir -p ~/BirdWatch/model && cd ~/BirdWatch/model
wget "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3?lite-format=tflite" -O birds_V1.tflite
wget https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv
```

### 4. Environment variables

Create `~/BirdWatch/.env`:

```env
BIRDWATCH_EMAIL=you@gmail.com
BIRDWATCH_EMAIL_PASS=your-gmail-app-password
BIRDWATCH_EMAIL_TO=you@gmail.com
```

> Gmail requires an App Password (Account → Security → App Passwords).
> Leave these unset if you don't want email alerts — the system works without them.

### 5. Run

```bash
# Terminal 1 — detection engine
python detector.py

# Terminal 2 — web UI
python app.py
# Visit http://<pi-ip>:5000
```

### 6. Auto-start on boot (systemd)

```bash
sudo cp scripts/birdwatch-detector.service /etc/systemd/system/
sudo cp scripts/birdwatch-app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable birdwatch-detector birdwatch-app
sudo systemctl start birdwatch-detector birdwatch-app
```

---

## Tuning

Key parameters in `detector.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MIN_CONTOUR_AREA` | 1200px | Minimum single contour to trigger |
| `MIN_TOTAL_AREA` | 2000px | Total motion area threshold |
| `SAVE_COOLDOWN_SEC` | 8s | Minimum time between detections |
| `BURST_MAX_FRAMES` | 8 | Max frames collected per visit |
| `BURST_TIMEOUT_SEC` | 3.0s | Stop burst after this much silence |
| `CROP_BOTTOM` | 0.82 | Ignore bottom 18% of frame |

Key parameters in `classifier.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CONFIDENCE_THRESHOLD` | 0.70 | Minimum confidence to accept classification |

---

## Checker

Re-run inference on existing detections with multi-pass averaging:

```bash
# Review all detections
python checker.py --passes 5

# Review only Unknown/background
python checker.py --only-unknown --passes 5

# Write corrections back to DB
python checker.py --passes 5 --update

# Test on a folder of images
python checker.py --folder ~/my_test_images --passes 5
```

---

## Training Your Own Model (Phase 5)

### 1. Download labeled training data

```bash
# Downloads photos from GBIF for Edmonton-area species
python download_training_data.py --limit 80

# Single species
python download_training_data.py --species "Pica hudsonia" --limit 60
```

### 2. Organize your own detections

```bash
# Copies high-confidence detections into species folders
python organize_dataset.py
```

### 3. Fine-tune on Google Colab

Upload the `dataset/` folder to Colab and run `scripts/colab_finetune.ipynb`.
The notebook fine-tunes MobileNetV2 via transfer learning and exports `birdwatch_custom.tflite`.

### 4. Deploy

```bash
scp birdwatch_custom.tflite salcxjo@<pi-ip>:~/BirdWatch/model/
scp birdwatch_labels.json   salcxjo@<pi-ip>:~/BirdWatch/model/
sudo systemctl restart birdwatch-detector
```

`classifier.py` automatically detects and uses the custom model if present.

---

## Future Considerations

- **Audio identification** — add a USB microphone and integrate BirdNET (Cornell Lab) for song-based species ID, which often outperforms vision alone
- **Servo pan/tilt tracking** — SG90 servos on a camera mount with PID control to follow detected birds across the frame
- **Fine-tuned custom classifier** — retrain MobileNetV2 on accumulated balcony footage for location-specific accuracy improvement
- **Confidence ensemble** — combine TFLite score with BirdNET audio score for higher-confidence multi-modal classification
- **eBird integration** — automatically submit confirmed sightings to eBird citizen science database
- **Temporal modeling** — use visit time patterns to improve species predictions (e.g. owls at night, waxwings in winter)
- **Weather correlation** — correlate visit frequency with temperature, pressure, wind using a weather API
- **Solar-powered deployment** — Pi 4 + battery + solar panel for rooftop or remote deployment without mains power
- **Web-accessible dashboard** — expose the Flask UI via Tailscale or Cloudflare Tunnel for remote access without port forwarding
- **Mobile push notifications** — replace email alerts with push notifications via Pushover or Ntfy
- **Species rarity alerts** — flag unusual species against expected Edmonton eBird frequency data

---

## Species Supported (Edmonton Filter)

The classifier suppresses non-local species. Recognized Edmonton-area birds include:
House Sparrow, American Robin, European Starling, Dark-eyed Junco, Downy Woodpecker,
Hairy Woodpecker, American Crow, Black-billed Magpie, Black-capped Chickadee,
White-breasted Nuthatch, House Finch, American Goldfinch, Chipping Sparrow,
Song Sparrow, White-crowned Sparrow, Northern Flicker, Cedar Waxwing, Bohemian Waxwing,
Rock Pigeon, Eurasian Collared-Dove, Canada Goose, Mallard, American Kestrel,
Cooper's Hawk, Red-tailed Hawk, Blue Jay, Common Grackle, Brown-headed Cowbird,
Yellow Warbler, Barred Owl, Great Horned Owl, and Osprey.

---

## License

MIT — free to use, modify, and deploy.

---

*Built by Salar Mirikhoozani — part of an ongoing series of embedded CV and edge ML projects.*
