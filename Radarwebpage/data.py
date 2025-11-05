import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
import tempfile
import pyart
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime as dt, timedelta
import gzip
import shutil

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
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
    if dest_dir is None:
        dest_dir = tempfile.gettempdir()
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
    return local_path


def ensure_uncompressed(path):
    """
    Detect gzip magic and, if gzipped, decompress to a new temp file and return its path.
    Otherwise return original path.
    """
    with open(path, "rb") as f:
        magic = f.read(2)
    # gzip magic bytes: 0x1f 0x8b
    if magic == b'\x1f\x8b':
        out_fd, out_path = tempfile.mkstemp(suffix=".raw")
        os.close(out_fd)
        with gzip.open(path, "rb") as gz, open(out_path, "wb") as out:
            shutil.copyfileobj(gz, out)
        print(f"Decompressed gzip {path} -> {out_path}")
        return out_path
    return path


def plot_level2_cartopy(local_file, out_png=None, title_station=None):
    if out_png is None:
        base = os.path.splitext(os.path.basename(local_file))[0]
        out_png = os.path.join(os.path.dirname(local_file), f"{base}_reflectivity_map.png")

    # Try reading; if failure due to compression, let caller handle decompression first
    radar = pyart.io.read_nexrad_archive(local_file)
    if 'reflectivity' in radar.fields:
        ref = radar.fields['reflectivity']['data']
    elif 'reflectivity_horizontal' in radar.fields:
        ref = radar.fields['reflectivity_horizontal']['data']
    else:
        raise RuntimeError("Reflectivity field not found in radar file.")

    ref = np.ma.masked_less(ref, 5)
    lats = radar.gate_latitude['data'].flatten()
    lons = radar.gate_longitude['data'].flatten()
    ref_flat = ref.flatten()
    valid = ~np.ma.getmaskarray(ref_flat)
    lats = lats[valid]; lons = lons[valid]; ref_flat = ref_flat[valid]

    # Guard: no valid points
    if ref_flat.size == 0:
        raise RuntimeError("No valid reflectivity points to plot.")

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=proj)
    try:
        min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
        min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
        pad_lon = (max_lon - min_lon) * 0.1
        pad_lat = (max_lat - min_lat) * 0.1
        ax.set_extent([min_lon - pad_lon, max_lon + pad_lon, min_lat - pad_lat, max_lat + pad_lat], crs=proj)
    except Exception:
        try:
            rlat = float(radar.latitude['data']); rlon = float(radar.longitude['data'])
            ax.set_extent([rlon - 2, rlon + 2, rlat - 2, rlat + 2], crs=proj)
        except Exception:
            pass
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # Use a valid colormap name. prefer 'NWSRef' if available, else fallback to 'viridis'
    try:
        cmap = plt.get_cmap('NWSRef')
    except Exception:
        cmap = plt.get_cmap('viridis')
    sc = ax.scatter(lons, lats, c=ref_flat, s=1, cmap=cmap, vmin=0, vmax=75, transform=proj)
    cb = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cb.set_label('Reflectivity (dBZ)')
    title_time = dt.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if title_station:
        plt.title(f"{title_station} Reflectivity — {title_time}")
    else:
        plt.title(f"Reflectivity — {title_time}")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved map to {out_png}")
    return out_png


if __name__ == "__main__":
    station = "KENX"
    key, bucket = find_latest_level2_key(station, days_back=5)
    if not key:
        raise SystemExit("No Level-II file found.")
    local = download_s3_object(bucket, key)
    # If pyart chokes on compression, try decompressing
    try:
        out = plot_level2_cartopy(local, title_station=station)
    except Exception as e:
        msg = str(e).lower()
        if "unknown compression record" in msg or "not a gzip file" in msg or "EOF marker" in msg:
            decompressed = ensure_uncompressed(local)
            out = plot_level2_cartopy(decompressed, title_station=station)
        else:
            print("Error reading/plotting radar file:", e)
            raise
    print("Done:", out)
