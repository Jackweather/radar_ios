import pyart
import fsspec
from metpy.plots import ctables
import matplotlib.pyplot as plt
import warnings
from datetime import datetime as dt
import numpy as np
import os
import json

warnings.filterwarnings("ignore")

output_dir = "static/radar"
os.makedirs(output_dir, exist_ok=True)

# Clear the output_dir folder by removing all files inside
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # remove file or link
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)  # remove directory recursively
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

datTime = dt.utcnow()
year = datTime.strftime("%Y")
month = datTime.strftime("%m")
day = datTime.strftime("%d")
hour = datTime.strftime("%H")
timeStr = f'{year}{month}{day}{hour}'

fs = fsspec.filesystem("s3", anon=True)

stations = ['KFCX']

for site in stations:
    pattern = f's3://noaa-nexrad-level2/{year}/{month}/{day}/{site}/{site}{year}{month}{day}_*'
    files = sorted(fs.glob(pattern), reverse=True)

    if not files:
        print(f"No files found for station {site}. Skipping...")
        continue

    latest_file = files[0]
    print(f"Processing station {site}: {latest_file}")

    radar = pyart.io.read_nexrad_archive(f's3://{latest_file}')

    cLat = radar.latitude['data']
    cLon = radar.longitude['data']
    print(f"Radar Location (Lat, Lon): ({cLat}, {cLon})")

    reflectivity_data = radar.fields['reflectivity']['data']
    # Mask values less than 5 dBZ
    reflectivity_data = np.ma.masked_less(reflectivity_data, 5)

    # --- REMOVE smoothing, use raw data directly ---
    radar.fields['reflectivity']['data'] = reflectivity_data

    ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(1, 1, 1)  # No map projection here!

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

    output_file = f"{output_dir}/{site}_Reflectivity_{timeStr}.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    print(f"Saved radar image to: {output_file}")

    gate_lats = radar.gate_latitude['data']
    gate_lons = radar.gate_longitude['data']

    flat_lats = gate_lats.flatten()
    flat_lons = gate_lons.flatten()

    min_lat = float(np.min(flat_lats))
    max_lat = float(np.max(flat_lats))
    min_lon = float(np.min(flat_lons))
    max_lon = float(np.max(flat_lons))

    bounds = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    }

    print(f"Geographic bounds of radar sweep for {site}:")
    print(bounds)

    bounds_file = f"{output_dir}/{site}_Reflectivity_{timeStr}_bounds.json"
    with open(bounds_file, 'w') as f:
        json.dump(bounds, f)

    print(f"Saved geographic bounds to: {bounds_file}")