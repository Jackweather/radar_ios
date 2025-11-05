import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for safe background plotting

import pyart
import fsspec
from metpy.plots import ctables
import matplotlib.pyplot as plt
import warnings
from datetime import datetime as dt, timedelta
import numpy as np
import json
import os
import glob
import threading
import time
from flask import Flask, render_template, jsonify, url_for, request
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import gzip
import shutil
import tempfile
import gc  # <--- add garbage-collection import

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variable to track radar generation status
generation_status = {"status": "Idle", "last_updated": ""}
# Global variable to track recently generated radar images
recent_images = []
# Flag to clear output folder on the first automated generation loop
first_run = True

# Add a lock to prevent race when multiple requests hit at once
radar_thread_lock = threading.Lock()

# Replace the previous download helpers and fs-based logic with boto3-based helpers
def find_latest_level2_key(station, days_back=3):
	"""
	Search unidata-nexrad-level2 bucket for the most recent full Level-II file
	for `station` across the last `days_back` days.
	Returns (key, bucket) or (None, bucket) if not found.
	"""
	s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
	bucket = 'unidata-nexrad-level2'
	found = []
	for d in range(days_back):
		date_dt = dt.utcnow() - timedelta(days=d)
		y = date_dt.strftime("%Y")
		m = date_dt.strftime("%m")
		day = date_dt.strftime("%d")
		prefix = f"{y}/{m}/{day}/{station}/"
		try:
			resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
		except Exception:
			continue
		if 'Contents' in resp:
			found.extend(resp['Contents'])
	if not found:
		print(f"No Level-II files found for {station} in the last {days_back} days.")
		return None, bucket
	latest = sorted(found, key=lambda x: x['LastModified'], reverse=True)[0]
	print(f"Most recent {station} file found: {latest['Key']}")
	return latest['Key'], bucket


def download_s3_object(bucket, key, dest_dir=None):
	"""
	Download a single S3 object using boto3 unsigned client to dest_dir (or temp dir).
	"""
	s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
	if dest_dir is None:
		dest_dir = tempfile.gettempdir()
	os.makedirs(dest_dir, exist_ok=True)
	local_path = os.path.join(dest_dir, os.path.basename(key))
	s3.download_file(bucket, key, local_path)
	print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
	return local_path


def ensure_uncompressed(path):
	"""Detect gzip magic and, if gzipped, attempt to decompress to a new temp file and return its path.
	If decompression fails we raise a RuntimeError with the original message so callers can skip that station.
	"""
	try:
		with open(path, "rb") as f:
			magic = f.read(2)
	except Exception as e:
		print(f"Warning opening file {path}: {e}")
		return path

	if magic == b'\x1f\x8b':
		fd, out_path = tempfile.mkstemp(suffix=".raw")
		os.close(fd)
		try:
			with gzip.open(path, "rb") as gz, open(out_path, "wb") as out:
				shutil.copyfileobj(gz, out)
			print(f"Decompressed gzip {path} -> {out_path}")
			return out_path
		except Exception as e:
			# Surface a clear, catchable error message for callers
			msg = str(e)
			print(f"Warning decompressing {path}: {msg}")
			# Keep original phrasing when possible so checks like "unknown compression record" match
			raise RuntimeError(msg)
	return path


def generate_radar_images():
	global generation_status, recent_images, first_run
	while True:
		try:
			generation_status["status"] = "Generating radar images..."
			generation_status["last_updated"] = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
			recent_images.clear()  # Clear the list of recent images for this cycle

			datTime = dt.utcnow()
			year = datTime.strftime("%Y")
			month = datTime.strftime("%m")
			day = datTime.strftime("%d")
			hour = datTime.strftime("%H")
			timeStr = f'{year}{month}{day}{hour}'

			stations = ['KLWX', 'KCCX', 'KJKL', 'KENX', 'KBGM', 'KTYX', 'KCXX', 'KBUF', 'KOKX', 'KFCX']
			output_dir = "static/radar"
			os.makedirs(output_dir, exist_ok=True)

			# On the first automated run, clear the output directory so it starts fresh
			if first_run:
				try:
					for fname in os.listdir(output_dir):
						path = os.path.join(output_dir, fname)
						if os.path.isfile(path) or os.path.islink(path):
							os.remove(path)
						elif os.path.isdir(path):
							shutil.rmtree(path)
					print(f"Cleared output directory: {output_dir}")
				except Exception as e:
					print(f"Warning clearing output directory {output_dir}: {e}")
				first_run = False

			for site in stations:
				try:
					# use boto3 search for most recent Level-II file for station
					key, bucket = find_latest_level2_key(site, days_back=3)
					if not key:
						print(f"No files found for station {site}. Skipping...")
						continue

					print(f"Processing station {site}: s3://{bucket}/{key}")

					# Download S3 object to local file before passing to pyart
					downloaded_file = download_s3_object(bucket, key, dest_dir=output_dir)  # renamed var
					# If gzipped, decompress before reading
					try:
						local_file = ensure_uncompressed(downloaded_file)
					except Exception as e:
						# ensure_uncompressed now raises RuntimeError on decompression problems
						errmsg = str(e).lower()
						print(f"Skipping station {site} due to decompression error: {e}")
						recent_images.append({
							"station": site,
							"image": "",
							"error": f"decompression error: {e}",
							"timestamp": timeStr
						})
						continue

					# Read from the local file
					radar = pyart.io.read_nexrad_archive(local_file)

					reflectivity_data = radar.fields['reflectivity']['data']
					reflectivity_data = np.ma.masked_less(reflectivity_data, 5)
					radar.fields['reflectivity']['data'] = reflectivity_data

					ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

					# Use smaller figure and lower DPI to reduce memory usage
					fig = plt.figure(figsize=(8, 8))
					ax = fig.add_subplot(1, 1, 1)

					display = pyart.graph.RadarMapDisplay(radar)

					display.plot_ppi(
						'reflectivity',
						0,
						ax=ax,
						cmap=ref_cmap,
						norm=ref_norm,
						vmin=15,
						vmax=75,
						colorbar_flag=False,
						title=''
					)

					ax.set_axis_off()
					fig.patch.set_alpha(0)
					ax.patch.set_alpha(0)

					output_file = f"{output_dir}/{site}.png"
					# save using the fig reference, lower dpi
					fig.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
					plt.close(fig)  # explicitly close the specific figure
					del fig, ax
					gc.collect()

					print(f"Saved radar image to: {output_file}")

					# Add the generated image to the recent images list (use leading slash for web path)
					recent_images.append({
						"station": site,
						"image": '/' + output_file.replace('\\', '/'),
						"timestamp": timeStr
					})

					gate_lats = radar.gate_latitude['data']
					gate_lons = radar.gate_longitude['data']

					flat_lats = gate_lats.flatten()
					flat_lons = gate_lons.flatten()

					bounds = {
						"min_lat": float(np.min(flat_lats)),
						"max_lat": float(np.max(flat_lats)),
						"min_lon": float(np.min(flat_lons)),
						"max_lon": float(np.max(flat_lons))
					}

					print(f"Geographic bounds of radar sweep for {site}: {bounds}")

					bounds_file = f"{output_dir}/{site}_bounds.json"
					with open(bounds_file, 'w') as f:
						json.dump(bounds, f)

					print(f"Saved geographic bounds to: {bounds_file}")

					# Clean up large objects and temporary files to free memory/disk
					try:
						del radar
						del reflectivity_data
						del gate_lats, gate_lons, flat_lats, flat_lons
					except Exception:
						pass
					gc.collect()
					# remove downloaded and decompressed files when present
					try:
						if os.path.exists(downloaded_file):
							os.remove(downloaded_file)
						# if ensure_uncompressed created a different file, remove it too
						if local_file != downloaded_file and os.path.exists(local_file):
							os.remove(local_file)
					except Exception as e:
						print(f"Warning cleaning temp files for {site}: {e}")
					gc.collect()

				except MemoryError as mem_err:
					# Skip station on MemoryError, attempt to free memory and continue
					print(f"MemoryError processing station {site}: {mem_err} -- skipping station")
					generation_status["status"] = "Idle"
					gc.collect()
					continue
				except Exception as station_err:
					# Log error for this station and continue with the next one
					errstr = str(station_err).lower()
					if "unknown compression record" in errstr or "decompression error" in errstr:
						print(f"Skipped station {site} due to compression/decompression issue: {station_err}")
					else:
						print(f"Error processing station {site}: {station_err}")
					recent_images.append({
						"station": site,
						"image": "",
						"error": str(station_err),
						"timestamp": timeStr
					})
					continue

			# mark idle when finished
			generation_status["status"] = "Idle"
		except Exception as e:
			# Treat known decompression/compression messages as non-fatal and continue loop
			msg = str(e).lower()
			if "unknown compression record" in msg or "decompression error" in msg:
				print(f"Non-fatal decompression error during generation cycle: {e} -- continuing to next cycle")
				generation_status["status"] = "Idle"
			else:
				generation_status["status"] = f"Error: {e}"
				print(f"Error in radar image generation: {e}")
		finally:
			# Ensure the function waits for 5 minutes before restarting
			time.sleep(5 * 60)


@app.route('/status')
def status():
    return jsonify(generation_status)


@app.route('/recent-images')
def recent_images_route():
    return jsonify(recent_images)


def get_latest_radar_image(station_id):
    radar_dir = os.path.join('static', 'radar')
    # pattern matches the new per-station single-file naming
    pattern = os.path.join(radar_dir, f"{station_id}.png")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return '/' + latest_file.replace('\\', '/')

def list_radar_images():
    radar_dir = os.path.join('static', 'radar')
    if not os.path.exists(radar_dir):
        return []
    files = [f for f in os.listdir(radar_dir) if f.lower().endswith('.png')]
    files.sort(reverse=True)
    return files


@app.route('/gallery')
def gallery():
    images = list_radar_images()
    return render_template('radar_gallery.html', images=images)


@app.route('/')
def index():
    stations = [
        {
            'id': 'KLWX',
            'lat': 38.97611237,
            'lon': -77.48750305,
            'bounds': [
                [34.84664293135119, -82.80239033783172],
                [43.10558009613012, -72.17260865799442]
            ],
        },
        {
            'id': 'KCCX',
            'lat': 40.92316818,
            'lon': -78.00372314,
            'bounds': [
                [36.7934961034473, -83.47281512714993],
                [45.052827879608266, -72.53464276292041]
            ],
        },
        {
            'id': 'KJKL',
            'lat': 37.59083176,
            'lon': -83.31305695,
            'bounds': [
                [33.46135540824945, -88.52720127885047],
                [41.72030030972457, -78.09891722192692]
            ],
        },
        {
            'id': 'KENX',
            'lat': 42.58655548,
            'lon': -74.06408691,
            'bounds': [
                [38.457080718757574, -79.67698036449032],
                [46.71602144898151, -68.45119092906985]
            ],
        },
        {
            'id': 'KBGM',
            'lat': 42.19969559,
            'lon': -75.98472595,
            'bounds': [
                [38.07022598117396, -81.5630192883354],
                [46.329154213920454, -70.40643144099836]
            ],
        },
        {
            'id': 'KTYX',
            'lat': 43.75569534,
            'lon': -75.67986298,
            'bounds': [
                [39.62602639385995, -81.4018462239696],
                [47.885361249069334, -69.95786833537181]
            ],
        },
        {
            'id': 'KCXX',
            'lat': 44.51100159,
            'lon': -73.16642761,
            'bounds': [
                [40.381133240418805, -78.962610989264],
                [48.64086714911371, -67.37025417510782]
            ],
        },
        {
            'id': 'KBUF',
            'lat': 42.94878769,
            'lon': -78.73677826,
            'bounds': [
                [38.8185378506761, -84.38376230288945],
                [47.07903820864557, -73.08978881930115]
            ],
        },
        {
            'id': 'KOKX',
            'lat': 40.86552811,
            'lon': -72.86391449,
            'bounds': [
                [36.73585731888025, -78.32822702710388],
                [44.99519571758983, -67.39961517721764]
            ],
        },
        {
              'id': 'KFCX',
              'lat': 37.0243988,
              'lon': -80.27397156,
              'bounds': [
                  [32.894726263234276, -85.44936594328767],
                  [41.15426188081905, -75.09859179072811]
    ],
}

        
    ]  

    for s in stations:
        latest_img = get_latest_radar_image(s['id'])
        s['image_url'] = latest_img if latest_img else ''

    return render_template("index.html", stations=stations)

@app.route('/run-generation', methods=['POST'])
def run_generation():
    global generation_status
    try:
        generation_status["status"] = "Manually triggered radar generation..."
        generation_status["last_updated"] = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        threading.Thread(target=generate_radar_images_once, daemon=True).start()
        return jsonify({"message": "Radar generation started manually."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_radar_images_once():
	global generation_status, recent_images
	try:
		recent_images.clear()  # Clear the list of recent images for this cycle

		datTime = dt.utcnow()
		year = datTime.strftime("%Y")
		month = datTime.strftime("%m")
		day = datTime.strftime("%d")
		hour = datTime.strftime("%H")
		timeStr = f'{year}{month}{day}{hour}'

		stations = ['KLWX', 'KCCX', 'KJKL', 'KENX', 'KBGM', 'KTYX', 'KCXX', 'KBUF', 'KOKX', 'KFCX']
		output_dir = "static/radar"
		os.makedirs(output_dir, exist_ok=True)

		for site in stations:
			try:
				# use boto3 search for most recent Level-II file for station
				key, bucket = find_latest_level2_key(site, days_back=3)
				if not key:
					print(f"No files found for station {site}. Skipping...")
					continue

				print(f"Processing station {site}: s3://{bucket}/{key}")

				# Download to local and pass local path to pyart
				downloaded_file = download_s3_object(bucket, key, dest_dir=output_dir)
				try:
					local_file = ensure_uncompressed(downloaded_file)
				except Exception as e:
					errmsg = str(e).lower()
					print(f"Skipping station {site} in one-shot run due to decompression error: {e}")
					recent_images.append({
						"station": site,
						"image": "",
						"error": f"decompression error: {e}",
						"timestamp": timeStr
					})
					continue

				radar = pyart.io.read_nexrad_archive(local_file)

				reflectivity_data = radar.fields['reflectivity']['data']
				reflectivity_data = np.ma.masked_less(reflectivity_data, 5)
				radar.fields['reflectivity']['data'] = reflectivity_data

				ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

				# Lower memory usage: smaller figure + lower DPI
				fig = plt.figure(figsize=(8, 8))
				ax = fig.add_subplot(1, 1, 1)

				display = pyart.graph.RadarMapDisplay(radar)

				display.plot_ppi(
					'reflectivity',
					0,
					ax=ax,
					cmap=ref_cmap,
					norm=ref_norm,
					vmin=15,
					vmax=75,
					colorbar_flag=False,
					title=''
				)

				ax.set_axis_off()
				fig.patch.set_alpha(0)
				ax.patch.set_alpha(0)

				output_file = f"{output_dir}/{site}.png"
				fig.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
				plt.close(fig)
				del fig, ax
				gc.collect()

				print(f"Saved radar image to: {output_file}")

				recent_images.append({
					"station": site,
					"image": '/' + output_file.replace('\\', '/'),
					"timestamp": timeStr
				})

				gate_lats = radar.gate_latitude['data']
				gate_lons = radar.gate_longitude['data']

				flat_lats = gate_lats.flatten()
				flat_lons = gate_lons.flatten()

				bounds = {
					"min_lat": float(np.min(flat_lats)),
					"max_lat": float(np.max(flat_lats)),
					"min_lon": float(np.min(flat_lons)),
					"max_lon": float(np.max(flat_lons))
				}

				print(f"Geographic bounds of radar sweep for {site}: {bounds}")

				bounds_file = f"{output_dir}/{site}_bounds.json"
				with open(bounds_file, 'w') as f:
					json.dump(bounds, f)

				print(f"Saved geographic bounds to: {bounds_file}")

			except Exception as station_err:
				errstr = str(station_err).lower()
				if "unknown compression record" in errstr or "decompression error" in errstr:
					print(f"Skipped station {site} in one-shot run due to compression/decompression issue: {station_err}")
				else:
					print(f"Error processing station {site} in one-shot run: {station_err}")
				recent_images.append({
					"station": site,
					"image": "",
					"error": str(station_err),
					"timestamp": timeStr
				})
				continue

		generation_status["status"] = "Idle"
		generation_status["last_updated"] = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
	except Exception as e:
		msg = str(e).lower()
		if "unknown compression record" in msg or "decompression error" in msg:
			print(f"Non-fatal decompression error during one-shot generation: {e} -- finishing one-shot run")
			generation_status["status"] = "Idle"
			generation_status["last_updated"] = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
		else:
			generation_status["status"] = f"Error: {e}"
			generation_status["last_updated"] = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
			print(f"Error in radar image generation: {e}")


# Add: helper to start the generator thread exactly once (per worker) and allow disabling via env var
def start_radar_thread_if_needed():
	# Honor environment variable to disable auto-start if desired (set to "0" or "false" to disable)
	start_opt = os.environ.get("START_RADAR_GENERATOR", "1").lower()
	if start_opt not in ("1", "true", "yes", "on"):
		print("Radar generator thread disabled via START_RADAR_GENERATOR environment variable.")
		return
	# Use lock to avoid races when many requests arrive simultaneously
	with radar_thread_lock:
		if not app.config.get("RADAR_THREAD_STARTED"):
			thread = threading.Thread(target=generate_radar_images, daemon=True)
			thread.start()
			app.config["RADAR_THREAD_STARTED"] = True
			print("Radar generator thread started.")
		else:
			# already started in this worker
			pass

# Replace before_first_request with before_request (works in all Flask/Werkzeug versions)
@app.before_request
def _start_thread():
	# lightweight: avoid starting generator for static files, favicon, and simple health/status requests
	path = (request.path or "").lower()
	if path.startswith('/static') or path == '/favicon.ico' or path in ('/status', '/recent-images', '/run-generation'):
		# skip starting on static or health-like endpoints
		return
	start_radar_thread_if_needed()


if __name__ == '__main__':
	# For local development only: start the generator thread and run Flask
	# (The thread starter is idempotent, so this won't double-start in the same process.)
	start_radar_thread_if_needed()

	# Run Flask app
	# Use gunicorn if deployed on Render or similar platforms
	import os
	port = int(os.environ.get("PORT", 5000))  # Render sets the PORT environment variable
	# Turn off debug reloader by default so we don't spawn extra processes that interfere with threads.
	debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
	app.run(host="0.0.0.0", port=port, debug=debug_mode)
