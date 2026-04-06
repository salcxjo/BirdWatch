# app.py — BirdWatch Web UI

from flask import Flask, render_template, jsonify, send_from_directory, Response
import sqlite3
import os
import requests
from functools import lru_cache

app = Flask(__name__)
DB_PATH = os.path.expanduser("~/BirdWatch/data/birdwatch.db")
DETECTIONS_DIR = os.path.expanduser("~/BirdWatch/detections/")

# Jinja2 filter
import os as _os
app.jinja_env.filters['basename'] = _os.path.basename

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Wikipedia image fetch ---
@lru_cache(maxsize=128)
def get_wiki_image(scientific_name):
    if not scientific_name or scientific_name.lower() in ("unknown", "background"):
        return None
    COMMON_NAMES = {
        "Pica hudsonia": "Black-billed magpie",
        "Corvus brachyrhynchos": "American crow",
        "Poecile atricapillus": "Black-capped chickadee",
        "Sitta carolinensis": "White-breasted nuthatch",
        "Haemorhous mexicanus": "House finch",
        "Spinus tristis": "American goldfinch",
        "Passer domesticus": "House sparrow",
        "Turdus migratorius": "American robin",
        "Junco hyemalis": "Dark-eyed junco",
        "Picoides pubescens": "Downy woodpecker",
        "Dryobates villosus": "Hairy woodpecker",
        "Colaptes auratus": "Northern flicker",
        "Bombycilla cedrorum": "Cedar waxwing",
        "Bombycilla garrulus": "Bohemian waxwing",
        "Columba livia": "Rock pigeon",
        "Sturnus vulgaris": "European starling",
        "Branta canadensis": "Canada goose",
        "Falco sparverius": "American kestrel",
        "Accipiter cooperii": "Cooper's hawk",
        "Buteo jamaicensis": "Red-tailed hawk",
    }
    query = COMMON_NAMES.get(scientific_name, scientific_name)
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
            headers={"User-Agent": "BirdWatch/1.0 (raspberry pi bird monitor)"},
            timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            url = data.get("thumbnail", {}).get("source")
            return url
    except Exception as e:
        print(f"Wiki error for {scientific_name}: {e}")
    return None

@app.route('/wiki-image/<path:scientific_name>')
def wiki_image(scientific_name):
    url = get_wiki_image(scientific_name)
    if not url:
        return "", 404
    try:
        img_data = requests.get(url, timeout=5).content
        return img_data, 200, {"Content-Type": "image/jpeg"}
    except Exception:
        return "", 404

# --- Latest detection image ---
@app.route('/latest')
def latest():
    conn = get_db()
    row = conn.execute(
        """SELECT image_path FROM detections
           WHERE species NOT IN ('Unknown','background')
           ORDER BY timestamp DESC LIMIT 1"""
    ).fetchone()
    conn.close()
    if row:
        return send_from_directory(DETECTIONS_DIR, os.path.basename(row['image_path']))
    return "No detections yet", 404

@app.route('/detections/<path:filename>')
def serve_detection(filename):
    return send_from_directory(DETECTIONS_DIR, filename)

# --- Pages ---
@app.route('/')
def dashboard():
    conn = get_db()
    recent = conn.execute("""
        SELECT * FROM detections
        WHERE species NOT IN ('Unknown','background')
        ORDER BY timestamp DESC LIMIT 12
    """).fetchall()
    species_counts = conn.execute("""
        SELECT species, COUNT(*) as count FROM detections
        WHERE species NOT IN ('Unknown','background')
        GROUP BY species ORDER BY count DESC
    """).fetchall()
    today_count = conn.execute("""
        SELECT COUNT(*) FROM detections
        WHERE date(timestamp,'localtime') = date('now','localtime')
        AND species NOT IN ('Unknown','background')
    """).fetchone()[0]
    total_species = conn.execute("""
        SELECT COUNT(DISTINCT species) FROM detections
        WHERE species NOT IN ('Unknown','background')
    """).fetchone()[0]
    conn.close()
    return render_template('dashboard.html',
                           recent=recent,
                           species_counts=species_counts,
                           today_count=today_count,
                           total_species=total_species)

@app.route('/gallery')
def gallery():
    conn = get_db()
    species_list = conn.execute("""
        SELECT species,
               COUNT(*) as count,
               MAX(timestamp) as last_seen,
               MAX(confidence) as best_confidence,
               image_path
        FROM detections
        WHERE species NOT IN ('Unknown','background')
        GROUP BY species ORDER BY count DESC
    """).fetchall()
    conn.close()
    return render_template('gallery.html', species_list=species_list)

@app.route('/logs')
def logs():
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM detections
        WHERE species NOT IN ('Unknown','background')
        ORDER BY timestamp DESC LIMIT 200
    """).fetchall()
    conn.close()
    return render_template('logs.html', rows=rows)

@app.route('/api/stats')
def api_stats():
    conn = get_db()
    hourly = conn.execute("""
        SELECT CAST(strftime('%H', timestamp, 'localtime') AS INTEGER) as hour,
               COUNT(*) as count
        FROM detections
        WHERE date(timestamp,'localtime') = date('now','localtime')
        AND species NOT IN ('Unknown','background')
        GROUP BY hour ORDER BY hour
    """).fetchall()
    all_time_hourly = conn.execute("""
        SELECT CAST(strftime('%H', timestamp, 'localtime') AS INTEGER) as hour,
               COUNT(*) as count
        FROM detections
        WHERE species NOT IN ('Unknown','background')
        GROUP BY hour ORDER BY hour
    """).fetchall()
    top_species = conn.execute("""
        SELECT species, COUNT(*) as count FROM detections
        WHERE species NOT IN ('Unknown','background')
        GROUP BY species ORDER BY count DESC LIMIT 10
    """).fetchall()
    conn.close()
    return jsonify({
        "hourly": [dict(r) for r in hourly],
        "all_time_hourly": [dict(r) for r in all_time_hourly],
        "top_species": [dict(r) for r in top_species]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
