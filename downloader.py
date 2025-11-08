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
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
import sys
import subprocess
from datetime import datetime, timedelta
from metpy.plots import ctables

# -------------------------------
# Settings
# -------------------------------
OUTPUT_DIR = os.path.join("static", "radar")
os.makedirs(OUTPUT_DIR, exist_ok=True)
STATIONS = ["KENX", "KOKX", "KBGM", "KBUF", "KTYX", "KBOX"]

# -------------------------------
# Helper Functions
# -------------------------------

def find_latest_level2_key(station, days_back=3):
    """Find latest NEXRAD Level-II file on AWS."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    bucket = "unidata-nexrad-level2"
    latest_key = None
    latest_mod = None

    for d in range(days_back):
        date_dt = datetime.utcnow() - timedelta(days=d)
        prefix = f"{date_dt:%Y/%m/%d}/{station}/"
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            for obj in resp.get("Contents", []):
                lm = obj.get("LastModified")
                if latest_mod is None or (lm and lm > latest_mod):
                    latest_mod = lm
                    latest_key = obj["Key"]
        except Exception:
            continue
    return latest_key, bucket


def download_s3_object(bucket, key, dest_dir):
    """Download file from S3 (unsigned)."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    local_path = os.path.join(dest_dir, os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path


def ensure_uncompressed(path):
    """Decompress .gz file if needed."""
    with open(path, "rb") as f:
        if f.read(2) == b"\x1f\x8b":
            out_path = tempfile.mktemp(suffix=".raw")
            with gzip.open(path, "rb") as gz, open(out_path, "wb") as out:
                shutil.copyfileobj(gz, out)
            return out_path
    return path


def process_station(station, output_dir):
    """Download, process, and save radar PNG + bounds JSON."""
    print(f"Processing {station}...")

    try:
        key, bucket = find_latest_level2_key(station)
        if not key:
            print(f"❌ No recent data for {station}")
            return

        tmp_dir = tempfile.mkdtemp()
        downloaded = download_s3_object(bucket, key, dest_dir=tmp_dir)
        local_file = ensure_uncompressed(downloaded)

        # read_nexrad_archive returns a Radar object (not a context manager)
        radar = pyart.io.read_nexrad_archive(local_file)
        try:
            if "reflectivity" not in radar.fields:
                print(f"⚠️ No reflectivity data for {station}")
                return

            refl = radar.fields["reflectivity"]["data"]
            radar.fields["reflectivity"]["data"] = np.ma.masked_less(refl, 5)
            ref_norm, ref_cmap = ctables.registry.get_with_steps("NWSReflectivity", 5, 5)

            # Plot (reduced size and DPI)
            fig, ax = plt.subplots(figsize=(6, 6))
            display = pyart.graph.RadarMapDisplay(radar)
            display.plot_ppi(
                "reflectivity", 0,
                ax=ax,
                cmap=ref_cmap,
                norm=ref_norm,
                vmin=15, vmax=75,
                colorbar_flag=False,
                title=""
            )
            ax.set_axis_off()
            fig.savefig(os.path.join(output_dir, f"{station}.png"),
                        dpi=80, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)

            # Calculate bounds efficiently
            bounds = {
                "min_lat": float(np.nanmin(radar.gate_latitude["data"])),
                "max_lat": float(np.nanmax(radar.gate_latitude["data"])),
                "min_lon": float(np.nanmin(radar.gate_longitude["data"])),
                "max_lon": float(np.nanmax(radar.gate_longitude["data"]))
            }
            with open(os.path.join(output_dir, f"{station}_bounds.json"), "w") as f:
                json.dump(bounds, f, indent=2)
        finally:
            # explicitly drop the radar reference so memory can be freed
            try:
                del radar
            except Exception:
                pass
            gc.collect()

        print(f"✅ {station} done")

    except Exception as e:
        print(f"⚠️ Error processing {station}: {e}")
    finally:
        # Full cleanup
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        plt.close("all")
        gc.collect()
        time.sleep(2)


# -------------------------------
# Main
# -------------------------------
def run_all(output_dir=OUTPUT_DIR):
    """Run each station in its own Python subprocess so memory is reclaimed on exit."""
    for stn in STATIONS:
        print(f"→ launching subprocess for {stn} ...", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--station", stn]
        try:
            # capture and print subprocess output so you still see progress in the parent terminal
            res = subprocess.run(cmd, cwd=os.path.dirname(__file__), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(res.stdout, end="", flush=True)
            if res.returncode != 0:
                print(f"⚠️ Subprocess for {stn} exited with code {res.returncode}", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to run subprocess for {stn}: {e}", flush=True)
        # brief pause between subprocesses
        time.sleep(1)
    print("\n🎉 All stations processed (each in its own subprocess).", flush=True)


if __name__ == "__main__":
    # CLI: allow running a single station in-process (used by the subprocess approach above)
    if len(sys.argv) >= 3 and sys.argv[1] == "--station":
        station_id = sys.argv[2]
        process_station(station_id, OUTPUT_DIR)
    else:
        run_all()
