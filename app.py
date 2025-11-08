from flask import Flask, render_template, jsonify, request, url_for, redirect
import os
import time
import base64
import datetime
import threading
import json

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

# Ensure downloader module is importable and expose run_all
import downloader

def list_radar_files():
    """Return list of radar files in static/radar sorted by mtime desc."""
    items = []
    try:
        files = [f for f in os.listdir(RADAR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    except FileNotFoundError:
        files = []
    files.sort(key=lambda f: os.path.getmtime(os.path.join(RADAR_DIR, f)), reverse=True)
    for fn in files:
        path = os.path.join(RADAR_DIR, fn)
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        # Determine station id robustly: basename without extension, take part before first underscore
        basename_no_ext = os.path.splitext(fn)[0]
        station = basename_no_ext.split('_')[0]
        # Determine corresponding bounds JSON (station_bounds.json)
        bounds_filename = f"{station}_bounds.json"
        bounds_path = os.path.join(RADAR_DIR, bounds_filename)
        bounds_exists = os.path.exists(bounds_path)
        bounds_url = url_for('static', filename='radar/' + bounds_filename) if bounds_exists else None
        items.append({
            'file': fn,
            'image': url_for('static', filename='radar/' + fn),
            'station': station,
            'timestamp': ts,
            'bounds_exists': bounds_exists,
            'bounds_url': bounds_url
        })
    return items

def latest_for_station(station_id):
    for item in list_radar_files():
        if item['station'] == station_id:
            return item['image']
    return None

def start_periodic_downloader(interval_seconds=300):
    """Start a background thread that runs downloader.run_all() every interval_seconds until all stations are present."""
    def worker():
        # Run immediately, then repeat until all stations have both PNG and bounds JSON
        while True:
            try:
                print("Periodic downloader: starting run_all()")
                downloader.run_all()
            except Exception as e:
                print(f"Periodic downloader error during run_all(): {e}")

            # Check whether each station has both PNG and bounds JSON in RADAR_DIR
            all_done = True
            for s in stations:
                png_path = os.path.join(RADAR_DIR, f"{s['id']}.png")
                json_path = os.path.join(RADAR_DIR, f"{s['id']}_bounds.json")
                if not (os.path.exists(png_path) and os.path.exists(json_path)):
                    all_done = False
                    break

            if all_done:
                print("Periodic downloader: all stations processed — stopping periodic runs.")
                return  # exit thread

            print(f"Periodic downloader: not complete yet, sleeping for {interval_seconds} seconds")
            time.sleep(interval_seconds)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

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

        json_path = os.path.join(RADAR_DIR, f"{s['id']}_bounds.json")
        try:
            with open(json_path, 'w') as jf:
                json.dump(bounds_obj, jf, indent=2)
        except Exception as e:
            print(f"Error writing bounds JSON for {s['id']}: {e}")

        created.append(fn)
    return jsonify({'message': f'Created {len(created)} placeholder images and bounds JSON.', 'files': created})

if __name__ == '__main__':
    # Start periodic downloader once (avoid double-start with Flask reloader)
    # When using debug mode, Werkzeug starts a child process; WERKZEUG_RUN_MAIN is set in the reloader child.
    start_thread = True
    if app.debug:
        # Only start in the reloader child process to avoid duplicated threads
        start_thread = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if start_thread:
        start_periodic_downloader(interval_seconds=300)

    # Run the app for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
