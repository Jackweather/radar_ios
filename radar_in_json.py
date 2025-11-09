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


def process_station(station, output_dir):
    """Download, process, and save radar data as JSON + bounds JSON."""
    # Process station (do not create the large {station}.json; only write location and bounds)
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

            # mask low values
            refl = radar.fields["reflectivity"]["data"]
            radar.fields["reflectivity"]["data"] = np.ma.masked_less(refl, 5)

            # helper: convert masked / NaN arrays to nested lists with None for JSON
            def arr_to_list(arr):
                # ensure numpy array
                a = np.array(arr)
                # if masked array, fill with np.nan so we can detect missing
                if np.ma.is_masked(a):
                    a = a.filled(np.nan)
                # handle any-dimensional arrays by converting to nested Python lists
                # Replace NaN with None (JSON null)
                if np.isnan(a).any():
                    # preserve shape; use vectorized approach for speed on 2D but simple python loops are clearer
                    py = []
                    for row in a:
                        py.append([None if (isinstance(x, float) and np.isnan(x)) else (float(x) if not isinstance(x, (list, np.ndarray)) else x) for x in row])
                    return py
                else:
                    return [[float(x) for x in row] for row in a]

            masked_refl = radar.fields["reflectivity"]["data"]  # masked array
            refl_list = arr_to_list(masked_refl)

            # Convert gate lat/lon (may be large) to lists
            lat_list = arr_to_list(radar.gate_latitude["data"])
            lon_list = arr_to_list(radar.gate_longitude["data"])

            # Calculate bounds efficiently
            bounds = {
                "min_lat": float(np.nanmin(radar.gate_latitude["data"])),
                "max_lat": float(np.nanmax(radar.gate_latitude["data"])),
                "min_lon": float(np.nanmin(radar.gate_longitude["data"])),
                "max_lon": float(np.nanmax(radar.gate_longitude["data"]))
            }

            # --- NEW: determine station lat/lon (center / instrument location) ---
            station_lat = None
            station_lon = None
            try:
                # preferred: radar.latitude['data'] / radar.longitude['data'] (may be scalar array)
                if hasattr(radar, 'latitude') and isinstance(radar.latitude, dict) and 'data' in radar.latitude:
                    station_lat = float(np.nanmean(radar.latitude['data']))
                elif 'latitude' in getattr(radar, 'metadata', {}):
                    station_lat = float(radar.metadata.get('latitude'))
                elif 'instrument_latitude' in getattr(radar, 'metadata', {}):
                    station_lat = float(radar.metadata.get('instrument_latitude'))
            except Exception:
                station_lat = None
            try:
                if hasattr(radar, 'longitude') and isinstance(radar.longitude, dict) and 'data' in radar.longitude:
                    station_lon = float(np.nanmean(radar.longitude['data']))
                elif 'longitude' in getattr(radar, 'metadata', {}):
                    station_lon = float(radar.metadata.get('longitude'))
                elif 'instrument_longitude' in getattr(radar, 'metadata', {}):
                    station_lon = float(radar.metadata.get('instrument_longitude'))
            except Exception:
                station_lon = None

            # ---- NEW VALIDATION: skip station if no valid location or bounds ----
            try:
                bounds_vals = [bounds.get("min_lat"), bounds.get("max_lat"), bounds.get("min_lon"), bounds.get("max_lon")]
                # require station lat/lon and all bounds to be finite numbers
                if station_lat is None or station_lon is None or not all(np.isfinite(bounds_vals)):
                    print(f"→ Skipping {station}: missing/invalid station location or bounds")
                    return
            except Exception:
                print(f"→ Skipping {station}: error validating location/bounds")
                return

            # Ensure output dir exists
            os.makedirs(output_dir, exist_ok=True)

            # DO NOT write the large {station}.json.
            # Only write small location and bounds files and the .rdat binary.
            try:
                loc = {"station": station, "lat": station_lat, "lon": station_lon}
                with open(os.path.join(output_dir, f"{station}_location.json"), "w") as lf:
                    json.dump(loc, lf, indent=2)
            except Exception as e:
                print(f"⚠️ Failed writing location JSON for {station}: {e}")

            # write bounds separately
            try:
                with open(os.path.join(output_dir, f"{station}_bounds.json"), "w") as f:
                    json.dump(bounds, f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed writing bounds JSON for {station}: {e}")

            # write compact binary .rdat for faster browser loading:
            def write_rdat(path, refl_arr, lat_arr, lon_arr, header_meta):
                # refl_arr, lat_arr, lon_arr expected as numpy arrays (2D) -- convert to float32, row-major
                rows, cols = refl_arr.shape
                header = dict(header_meta)
                header.update({"rows": int(rows), "cols": int(cols), "dtype": "float32", "order": "C"})
                header_bytes = json.dumps(header).encode("utf-8")
                with open(path, "wb") as f:
                    # write 4-byte little-endian header length, then header bytes, then raw float32 bytes
                    f.write(struct.pack("<I", len(header_bytes)))
                    f.write(header_bytes)
                    # ensure contiguous C-order float32
                    f.write(np.ascontiguousarray(refl_arr.astype(np.float32)).tobytes(order="C"))
                    f.write(np.ascontiguousarray(lat_arr.astype(np.float32)).tobytes(order="C"))
                    f.write(np.ascontiguousarray(lon_arr.astype(np.float32)).tobytes(order="C"))

            # Prepare arrays (use filled arrays so masked values become NaN) and write .rdat
            try:
                refl_np = np.array(masked_refl.filled(np.nan))  # masked_refl defined earlier
                lat_np = np.array(radar.gate_latitude["data"])
                lon_np = np.array(radar.gate_longitude["data"])
                rdat_path = os.path.join(output_dir, f"{station}.rdat")
                rdat_header = {
                    "station": station,
                    "bounds": bounds,
                    "time_units": radar.time.get("units"),
                    "metadata": {"radar_name": radar.metadata.get("instrument_name", None)}
                }
                write_rdat(rdat_path, refl_np, lat_np, lon_np, rdat_header)
            except Exception as e:
                print(f"⚠️ Failed writing binary .rdat for {station}: {e}")
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
        gc.collect()
        time.sleep(2)


# -------------------------------
# Main
# -------------------------------
def run_all(output_dir=OUTPUT_DIR):
    """Run each station sequentially in-process (no subprocesses, no delay)."""
    for stn in STATIONS:
        print(f"→ processing {stn} ...", flush=True)
        try:
            process_station(stn, output_dir)
        except Exception as e:
            print(f"⚠️ Error processing {stn}: {e}", flush=True)
    print("\n🎉 All stations processed.", flush=True)


if __name__ == "__main__":
    # CLI: allow running a single station in-process (used by the subprocess approach above)
    if len(sys.argv) >= 3 and sys.argv[1] == "--station":
        station_id = sys.argv[2]
        process_station(station_id, OUTPUT_DIR)
    else:
        run_all()
