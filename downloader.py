import os
import tempfile
import shutil
import gzip
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyart
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend to reduce memory usage
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from datetime import datetime, timedelta
from metpy.plots import ctables

# -------------------------------
# Settings
# -------------------------------
# Put all generated files directly under this base directory
BASE_DIR = '/var/data'
os.makedirs(BASE_DIR, exist_ok=True)
# For backward-compatibility use OUTPUT_DIR variable elsewhere in the file
OUTPUT_DIR = BASE_DIR

# List of radar stations to process (you can edit this)
STATIONS = ["KENX", "KOKX", "KBGM", "KBUF", "KTYX",'KBOX']

# -------------------------------
# Helper functions
# -------------------------------

def find_latest_level2_key(station, days_back=3):
    """Find latest radar Level-II file on AWS for given station."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
    bucket = 'unidata-nexrad-level2'
    latest_key = None
    latest_mod = None
    for d in range(days_back):
        date_dt = datetime.utcnow() - timedelta(days=d)
        y = date_dt.strftime("%Y")
        m = date_dt.strftime("%m")
        day = date_dt.strftime("%d")
        prefix = f"{y}/{m}/{day}/{station}/"
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in resp:
                for obj in resp['Contents']:
                    lm = obj.get('LastModified')
                    if latest_mod is None or (lm and lm > latest_mod):
                        latest_mod = lm
                        latest_key = obj['Key']
                # no need to keep all items in memory; just track the latest
        except Exception:
            continue
    return latest_key, bucket


def download_s3_object(bucket, key, dest_dir):
    """Download file from S3 (unsigned access)."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path


def ensure_uncompressed(path):
    """Decompress .gz file if needed."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        fd, out_path = tempfile.mkstemp(suffix=".raw")
        os.close(fd)
        with gzip.open(path, "rb") as gz, open(out_path, "wb") as out:
            shutil.copyfileobj(gz, out)
        return out_path
    return path


def process_station(station, output_dir):
    """Download, process, and save radar PNG and bounds JSON."""
    print(f"Processing {station}...")
    try:
        key, bucket = find_latest_level2_key(station)
        if not key:
            print(f"❌ No files found for {station}")
            return

        downloaded = download_s3_object(bucket, key, dest_dir=output_dir)
        local_file = ensure_uncompressed(downloaded)

        radar = pyart.io.read_nexrad_archive(local_file)

        # Mask weak reflectivity values (in-place where possible)
        if 'reflectivity' in radar.fields:
            refl = radar.fields['reflectivity']['data']
            # perform masking in a way that avoids allocating a large extra flattened copy
            radar.fields['reflectivity']['data'] = np.ma.masked_less(refl, 5)

        ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

        # Plot radar (slightly lower DPI to reduce memory during rendering)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        display = pyart.graph.RadarMapDisplay(radar)
        display.plot_ppi(
            'reflectivity', 0,
            ax=ax,
            cmap=ref_cmap,
            norm=ref_norm,
            vmin=15, vmax=75,
            colorbar_flag=False,
            title=''
        )
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        png_path = os.path.join(output_dir, f"{station}.png")
        fig.savefig(png_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        plt.close('all')  # ensure all figures cleared

        # Compute geographic bounds without flattening (avoid creating large copies)
        gate_lats = radar.gate_latitude['data']
        gate_lons = radar.gate_longitude['data']

        bounds = {
            "min_lat": float(np.min(gate_lats)),
            "max_lat": float(np.max(gate_lats)),
            "min_lon": float(np.min(gate_lons)),
            "max_lon": float(np.max(gate_lons))
        }

        # write bounds JSON into the same output directory as the PNG
        json_path = os.path.join(output_dir, f"{station}_bounds.json")
        with open(json_path, "w") as f:
            json.dump(bounds, f, indent=2)

        # Print both saved file locations (normalized absolute paths) so they are clear in the terminal
        png_display = os.path.normpath(os.path.abspath(png_path))
        json_display = os.path.normpath(os.path.abspath(json_path))
        print(f"✅ {station} complete — png: {png_display}  json: {json_display}", flush=True)

        # Cleanup: remove files and free large objects explicitly
        try:
            os.remove(downloaded)
            if downloaded != local_file:
                os.remove(local_file)
        except Exception:
            pass

        # aggressively drop references and run GC
        try:
            for _name in ('radar', 'display', 'gate_lats', 'gate_lons', 'refl', 'fig', 'ax'):
                if _name in locals():
                    locals().pop(_name, None)
        except Exception:
            pass
        plt.close('all')
        gc.collect()
        gc.collect()  # run twice to be sure large objects are reclaimed

    except Exception as e:
        print(f"⚠️ Error processing {station}: {e}")


# -------------------------------
# Main loop
# -------------------------------
def run_all(output_dir=OUTPUT_DIR):
    """Run processing for all stations once."""
    for stn in STATIONS:
        process_station(stn, output_dir)
        # ensure we pause between stations in the main loop
        print("⏳ waiting 3s before next station...", flush=True)
        time.sleep(3)
        gc.collect()
    print("\n🎉 All stations processed! PNG + JSON saved in static/radar/")
if __name__ == "__main__":
    # When executed directly, run once= "__main__":
    run_all()    # When executed directly, run once

    run_all()
