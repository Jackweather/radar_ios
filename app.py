from flask import Flask, render_template, jsonify, request, url_for, redirect, send_from_directory
import os
import time
import base64
import datetime
import json
import subprocess
import threading
import traceback
import getpass
import sys

app = Flask(__name__, static_folder='static', template_folder='templates')

# Ensure radar static directory exists
RADAR_DIR = os.path.join(os.path.dirname(__file__), 'static', 'radar')
os.makedirs(RADAR_DIR, exist_ok=True)

# Minimal set of stations used by templates (id, lat, lon, bounds)
stations = [
    {'id': 'KTYX', 'lat': 44.3, 'lon': -71.3, 'bounds': [[44.0, -71.7], [44.6, -70.9]]},
    {'id': 'KRLX', 'lat': 38.9, 'lon': -80.2, 'bounds': [[38.6, -80.6], [39.2, -79.8]]},
    {'id': 'KOKX', 'lat': 40.9, 'lon': -73.7, 'bounds': [[40.5, -74.1], [41.3, -73.3]]},
]

# A small 1x1 transparent PNG (base64) used as a placeholder image
_PLACEHOLDER_PNG = base64.b64decode(
    b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
)

def list_radar_files():
    """Return list of radar files in static/radar sorted by mtime desc."""
    items = []
    try:
        # include all regular files in the radar directory except the index/list helpers
        files = [
            f for f in os.listdir(RADAR_DIR)
            if os.path.isfile(os.path.join(RADAR_DIR, f)) and f not in ('index.html', 'list.json')
        ]
    except FileNotFoundError:
        files = []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(RADAR_DIR, f)), reverse=True)
    for fn in files:
        path = os.path.join(RADAR_DIR, fn)
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        # station guess: part before first underscore or whole basename
        basename_no_ext = os.path.splitext(fn)[0]
        station = basename_no_ext.split('_')[0] if '_' in basename_no_ext else basename_no_ext
        ext = os.path.splitext(fn)[1].lower()
        is_image = ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')
        # station-level bounds
        bounds_filename = f"{station}_bounds.json"
        bounds_path = os.path.join(RADAR_DIR, bounds_filename)
        bounds_exists = os.path.exists(bounds_path)
        bounds_url = url_for('static', filename='radar/' + bounds_filename) if bounds_exists else None
        items.append({
            'file': fn,
            'image': url_for('static', filename='radar/' + fn),
            'station': station,
            'timestamp': ts,
            'ext': ext,
            'is_image': is_image,
            'bounds_exists': bounds_exists,
            'bounds_url': bounds_url
        })
    return items

def latest_for_station(station_id):
    for item in list_radar_files():
        if item['station'] == station_id:
            return item['image']
    return None

@app.route('/', endpoint='index')
def root():
    # Try serving index from templates first (preferred location)
    try:
        # look for a few possible template names
        tpl_candidates = ['radar_index.html', 'index.html']
        for tpl in tpl_candidates:
            tpl_path = os.path.join(app.template_folder or 'templates', tpl)
            if os.path.exists(tpl_path):
                return render_template(tpl)
    except Exception:
        pass

    # Next try a project-root index.html (if you moved the file to repo root)
    try:
        root_index = os.path.join(os.path.dirname(__file__), 'index.html')
        if os.path.exists(root_index):
            return send_from_directory(os.path.dirname(__file__), 'index.html')
    except Exception:
        pass

    # Then try serving the legacy static/radar/index.html (backwards-compatible)
    try:
        return send_from_directory(RADAR_DIR, 'index.html')
    except Exception:
        # Fallback: preserve previous behavior if none of the static/index/template files are available
        images = list_radar_files()
        return render_template('radar_gallery.html', images=images)

# Replace the previous index() function with a simple redirect to the named endpoint
@app.route('/index')
def index_redirect():
    # Redirect to the 'index' endpoint (which is '/')
    return redirect(url_for('index'))

@app.route('/radar.html')
def radar_html():
    # Alias for explicit radar.html path
    return root()

@app.route('/gallery')
def gallery():
    # Alias for /gallery as well
    return root()

@app.route('/recent-images')
def recent_images():
    # Return up to 10 recent images (most recent first)
    return jsonify(list_radar_files()[:10])

@app.route('/status')
def status():
    files = list_radar_files()
    last = files[0]['timestamp'] if files else ''
    return jsonify({'status': 'idle', 'last_updated': last})

@app.route('/run-generation', methods=['POST'])
def run_generation():
    # Create placeholder images for each station (filename: {ID}_{timestamp}.png)
    ts = int(time.time())
    created = []
    for s in stations:
        fn = f"{s['id']}_{ts}.png"
        path = os.path.join(RADAR_DIR, fn)
        with open(path, 'wb') as fh:
            fh.write(_PLACEHOLDER_PNG)

        # create or update a station-level bounds JSON so the gallery can link to it
        # use the bounds provided in the stations list if available
        b = s.get('bounds', None)
        if b and isinstance(b, list) and len(b) >= 2:
            bounds_obj = {
                "min_lat": float(b[0][0]),
                "min_lon": float(b[0][1]),
                "max_lat": float(b[1][0]),
                "max_lon": float(b[1][1]),
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "png": fn
            }
        else:
            bounds_obj = {
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "png": fn
            }

        json_path = os.path.join(RADAR_DIR, f"{s['id']}_bounds.json")
        try:
            with open(json_path, 'w') as jf:
                json.dump(bounds_obj, jf, indent=2)
        except Exception as e:
            print(f"Error writing bounds JSON for {s['id']}: {e}")

        created.append(fn)
    return jsonify({'message': f'Created {len(created)} placeholder images and bounds JSON.', 'files': created})

@app.route('/run-task1', methods=['POST', 'GET'])
def run_task1():
    """Start radar_in_json.py in a background thread and return immediately."""
    def run_all_scripts():
        try:
            user = getpass.getuser()
        except Exception:
            user = 'unknown'
        print(f"Background task: running radar_in_json.py as user: {user}")

        # point to radar_in_json.py instead of downloader.py
        radar_script = os.path.join(os.path.dirname(__file__), 'radar_in_json.py')
        cwd = os.path.dirname(__file__)
        if not os.path.exists(radar_script):
            print(f"Background task error: radar_in_json.py not found at {radar_script}")
            return

        try:
            print(f"Running radar_in_json.py (cwd={cwd})")
            result = subprocess.run(
                ["python", radar_script],
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("radar_in_json.py ran successfully.")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running radar_in_json.py: returncode={getattr(e,'returncode','unknown')}")
            print("STDOUT:", getattr(e, 'stdout', ''))
            print("STDERR:", getattr(e, 'stderr', ''))
            print(traceback.format_exc())
        except Exception as ex:
            print(f"Unexpected error running radar_in_json.py: {ex}")
            print(traceback.format_exc())

    threading.Thread(target=run_all_scripts, daemon=True).start()
    return ("Task started in background. Check server logs for output.", 202)

if __name__ == '__main__':
    # Run the app for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
