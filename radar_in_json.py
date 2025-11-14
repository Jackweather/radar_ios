import os
import tempfile
import shutil
import gzip
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyart
# try to import pyart ctables registry for official NWS colormap
try:
    from pyart import ctables  # preferred location
    _HAS_CT = True
except Exception:
    try:
        from pyart.graph import ctables  # alternate location in some versions
        _HAS_CT = True
    except Exception:
        ctables = None
        _HAS_CT = False
import numpy as np
import gc
import time
import sys
from datetime import datetime, timedelta
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# try to import scipy for high-quality interpolation/smoothing; fall back gracefully
try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -------------------------------
# Settings
# -------------------------------
OUTPUT_DIR = os.path.join("static", "radar")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ensure KBUF is included
STATIONS = ["KBUF","KCXX", "KDVN", "KENX", "KOKX", "KBGM",  "KTYX", "KBOX"]

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


# add helper to create a realistic NWS reflectivity colormap
def create_nws_colormap():
	"""
	Return a ListedColormap that approximates the NWS reflectivity palette.
	Masked/NaN values are set fully transparent.
	"""
	from matplotlib.colors import ListedColormap
	# Rough NWS-like palette from low->high dBZ
	colors = [
		"#9cc4ff",  # very light blue
		"#0096ff",  # blue
		"#00e600",  # green-cyan
		"#33ff33",  # green
		"#ffff00",  # yellow
		"#ffbf00",  # amber
		"#ff8000",  # orange
		"#ff4000",  # deep orange
		"#ff0000",  # red
		"#b0007f",  # magenta
		"#800080",  # purple
		"#ffffff",  # white (extreme)
	]
	cmap = ListedColormap(colors, name="custom_NWSRef")
	# make masked/NaN values fully transparent
	try:
		cmap.set_bad((0.0, 0.0, 0.0, 0.0))
	except Exception:
		# some matplotlib versions may not support set_bad on ListedColormap
		pass
	return cmap


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

        # Load radar (only reflectivity). If Py-ART raises an OSError about unknown compression,
        # skip this station and continue to the next one.
        try:
            radar = pyart.io.read_nexrad_archive(local_file, include_fields=["reflectivity"])
        except OSError as e:
            msg = str(e).lower()
            if "unknown compression" in msg or "unknown compression record" in msg:
                print(f"→ Skipping {station}: unknown compression ({e})")
                return
            raise

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

        # Render reflectivity (sweep 0) to PNG using high-quality (no interpolation) rendering
        try:
            sweep = 0
            sl = radar.get_slice(sweep)  # slice of rays for this sweep
            # select sweep data
            sweep_refl = refl[sl, :].copy()
            sweep_lat = lat[sl, :].copy()
            sweep_lon = lon[sl, :].copy()

            # mask invalid so background is transparent
            sweep_refl = np.ma.masked_invalid(sweep_refl)

            # plotting params for high quality without interpolation
            FIGSIZE = (12, 12)
            DPI = 700
            VMIN, VMAX = 0, 80

            fig, ax = plt.subplots(figsize=FIGSIZE)

            # Prefer Py-ART's official NWS reflectivity table if available;
            # fall back to the local create_nws_colormap() implementation.
            if _HAS_CT:
                try:
                    ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)
                except Exception:
                    ref_norm, ref_cmap = None, create_nws_colormap()
            else:
                ref_norm, ref_cmap = None, create_nws_colormap()
            # ensure masked values transparent if colormap supports it
            try:
                ref_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            except Exception:
                pass

            # Use normalization from Py-ART if provided; otherwise use vmin/vmax
            if ref_norm is not None:
                pcm = ax.pcolormesh(
                    sweep_lon, sweep_lat, sweep_refl,
                    cmap=ref_cmap, norm=ref_norm, shading="auto", rasterized=True
                )
            else:
                pcm = ax.pcolormesh(
                    sweep_lon, sweep_lat, sweep_refl,
                    cmap=ref_cmap, vmin=VMIN, vmax=VMAX, shading="auto", rasterized=True
                )

            # remove axes, ticks and ensure transparent background
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_facecolor("none")

            png_path = os.path.join(output_dir, f"{station}.png")
            plt.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Could not render PNG for {station}: {e}")

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

