import os
import tempfile
import shutil
import gzip
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyart
import numpy as np
import gc
import time
import sys
from datetime import datetime, timedelta
import struct

# -------------------------------
# Settings
# -------------------------------
OUTPUT_DIR = os.path.join("static", "radar")
os.makedirs(OUTPUT_DIR, exist_ok=True)
STATIONS = ["KDVN", "KENX", "KOKX", "KBGM", "KBUF", "KTYX", "KBOX"]

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


def write_rdat(path, refl_arr, lat_arr, lon_arr, header_meta):
    """Write compact binary .rdat file."""
    rows, cols = refl_arr.shape
    header = dict(header_meta)
    header.update({"rows": int(rows), "cols": int(cols), "dtype": "float32", "order": "C"})
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(np.ascontiguousarray(refl_arr.astype(np.float32)).tobytes(order="C"))
        f.write(np.ascontiguousarray(lat_arr.astype(np.float32)).tobytes(order="C"))
        f.write(np.ascontiguousarray(lon_arr.astype(np.float32)).tobytes(order="C"))


def process_station(station, output_dir):
    """Download, process, and save radar data as JSON + .rdat (memory optimized)."""
    print(f"Processing {station}...")

    tmp_dir = tempfile.mkdtemp()
    try:
        key, bucket = find_latest_level2_key(station)
        if not key:
            print(f"❌ No recent data for {station}")
            return

        downloaded = download_s3_object(bucket, key, dest_dir=tmp_dir)
        local_file = ensure_uncompressed(downloaded)

        # Load radar (only reflectivity)
        radar = pyart.io.read_nexrad_archive(local_file, include_fields=["reflectivity"])

        if "reflectivity" not in radar.fields:
            print(f"⚠️ No reflectivity data for {station}")
            del radar
            gc.collect()
            return

        # Extract arrays as float32 to save memory
        refl = radar.fields["reflectivity"]["data"].astype(np.float32)
        refl = np.ma.masked_less(refl, 5).filled(np.nan)
        lat = radar.gate_latitude["data"].astype(np.float32)
        lon = radar.gate_longitude["data"].astype(np.float32)

        # Compute bounds quickly
        bounds = {
            "min_lat": float(np.nanmin(lat)),
            "max_lat": float(np.nanmax(lat)),
            "min_lon": float(np.nanmin(lon)),
            "max_lon": float(np.nanmax(lon))
        }

        # Station lat/lon
        station_lat = float(np.nanmean(radar.latitude["data"])) if "data" in radar.latitude else None
        station_lon = float(np.nanmean(radar.longitude["data"])) if "data" in radar.longitude else None

        # Skip invalid
        if (station_lat is None or station_lon is None or
            not np.isfinite(station_lat) or not np.isfinite(station_lon)):
            print(f"→ Skipping {station}: invalid coordinates")
            del radar
            gc.collect()
            return

        # Write small JSONs
        loc = {"station": station, "lat": station_lat, "lon": station_lon}
        with open(os.path.join(output_dir, f"{station}_location.json"), "w") as lf:
            json.dump(loc, lf, indent=2)
        with open(os.path.join(output_dir, f"{station}_bounds.json"), "w") as bf:
            json.dump(bounds, bf, indent=2)

        # Write binary .rdat file
        rdat_header = {
            "station": station,
            "bounds": bounds,
            "time_units": radar.time.get("units"),
            "metadata": {"radar_name": radar.metadata.get("instrument_name", None)}
        }
        rdat_path = os.path.join(output_dir, f"{station}.rdat")
        write_rdat(rdat_path, refl, lat, lon, rdat_header)

        # Explicit cleanup
        del radar, refl, lat, lon
        gc.collect()
        print(f"✅ {station} done")

    except Exception as e:
        print(f"⚠️ Error processing {station}: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        time.sleep(1.5)


def run_all(output_dir=OUTPUT_DIR):
    """Run all stations sequentially with full cleanup between."""
    for stn in STATIONS:
        print(f"→ Processing {stn} ...", flush=True)
        process_station(stn, output_dir)
        gc.collect()
        time.sleep(1)
    print("\n🎉 All stations processed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--station":
        station_id = sys.argv[2]
        process_station(station_id, OUTPUT_DIR)
    else:
        run_all()
