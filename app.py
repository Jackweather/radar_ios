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

app = Flask(__name__, static_folder='static', template_folder='templates')

# Base directory where downloader now writes files
BASE_DIR = '/var/data'
os.makedirs(BASE_DIR, exist_ok=True)
# App will list/serve images and JSON from BASE_DIR
RADAR_DIR = BASE_DIR

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
    """Return list of radar files in RADAR_DIR sorted by mtime desc."""
    items = []
    try:
        files = [f for f in os.listdir(RADAR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    except FileNotFoundError:
        files = []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(RADAR_DIR, f)), reverse=True)
    for fn in files:
        path = os.path.join(RADAR_DIR, fn)
        mtime = int(os.path.getmtime(path))
        ts = datetime.datetime.fromtimestamp(mtime).isoformat()
        # Determine station id robustly: basename without extension, take part before first underscore
        basename_no_ext = os.path.splitext(fn)[0]
        station = basename_no_ext.split('_')[0]
        # Determine corresponding bounds JSON (station_bounds.json)
        bounds_filename = f"{station}_bounds.json"
        bounds_path = os.path.join(RADAR_DIR, bounds_filename)
        bounds_exists = os.path.exists(bounds_path)
        # Append cache-busting query param using file mtime so browser reloads when file changes
        image_url = url_for('data_file', filename=fn) + f"?v={mtime}"
        bounds_url = (url_for('data_file', filename=bounds_filename) + f"?v={int(os.path.getmtime(bounds_path))}") if bounds_exists else None
        items.append({
            'file': fn,
            'image': image_url,
            'station': station,
            'timestamp': ts,
            'bounds_exists': bounds_exists,
            'bounds_url': bounds_url,
            'mtime': mtime
        })
    return items

@app.route('/data/<path:filename>')
def data_file(filename):
    """Serve files from BASE_DIR (where downloader writes PNG/JSON). Disable caching."""
    return send_from_directory(RADAR_DIR, filename, as_attachment=False, cache_timeout=0)

def latest_for_station(station_id):
    for item in list_radar_files():
        if item['station'] == station_id:
            return item['image']
    return None

@app.route('/', endpoint='index')
def root():
    # Serve the gallery as the root page (radar.html)
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

        # write generated bounds JSON into the same RADAR_DIR so the gallery can find it
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
    """Start downloader.py in a background thread and return immediately."""
    def run_all_scripts():
        try:
            user = getpass.getuser()
        except Exception:
            user = 'unknown'
        print(f"Background task: running scripts as user: {user}")

        # Replace the hardcoded scripts list with a portable downloader invocation
        downloader_path = os.path.join(os.path.dirname(__file__), 'downloader.py')
        cwd = os.path.dirname(__file__)
        if not os.path.exists(downloader_path):
            print(f"Background task error: downloader.py not found at {downloader_path}")
            return

        try:
            print(f"Running downloader.py (cwd={cwd})")
            result = subprocess.run(
                ["python", downloader_path],
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("downloader.py ran successfully.")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running downloader.py: returncode={getattr(e,'returncode','unknown')}")
            print("STDOUT:", getattr(e, 'stdout', ''))
            print("STDERR:", getattr(e, 'stderr', ''))
            print(traceback.format_exc())
        except Exception as ex:
            print(f"Unexpected error running downloader.py: {ex}")
            print(traceback.format_exc())

    threading.Thread(target=run_all_scripts, daemon=True).start()
    return ("Task started in background. Check server logs for output.", 202)

if __name__ == '__main__':
    # Run the app for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
