"""LULC prototype: Sentinel-1 -> 12 monthly land use / land cover maps for any location.

Input : latitude, longitude and a start month (YYYY-MM)
Steps : 1. retrieve Sentinel-1 data from Google Earth Engine and build 12 monthly composites
           (same preprocessing as the training data)
        2. run the trained U-Net on every month -> 12 LULC maps
        3. calculate the % of area per class, the month-to-month change and the total change
           (first -> last month), both in percentage points
        4. visualise the 12 maps and the class percentages over time
        5. optional: compare with Google Dynamic World (Sentinel-2 based, NOT used in training)
           as an independent sanity check

Example:
    python lulc_prototype.py --lat 49.4875 --lon 8.4660 --start 2025-01 \
        --project lulc-bharat-2026 --weights model_best.pt --config model_config.json --out results_mannheim
"""
import argparse
import json
import os

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from model import build_model

IGNORE = 255

# Dynamic World classes -> our 7 classes
# DW: 0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops, 5 shrub_and_scrub, 6 built, 7 bare, 8 snow_and_ice
# Ours: 0 Tree, 1 Grass/Shrub, 2 Cropland, 3 Built-up, 4 Bare/Sparse, 5 Water, 6 Wetland
DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_TO = [5, 0, 1, 6, 2, 1, 3, 4, 4]


# ---------------------------------------------------------------- 1. data retrieval
def month_list(start, n=12):
    """'2025-01' -> ['2025-01', ..., '2025-12']"""
    return [str(p) for p in pd.period_range(start=start, periods=n, freq='M')]


def s1_monthly_composite(region, month):
    """Median of all Sentinel-1 IW scenes in one month. Identical to Notebook 1:
    GEE S1_GRD is already calibrated, terrain-corrected and in dB; we add border-noise masking,
    a 3x3 focal-median speckle filter and the VV-VH band."""
    start = ee.Date(month + '-01')
    end = start.advance(1, 'month')

    def clean(img):
        img = img.updateMask(img.select('VV').gt(-30))
        return img.focal_median(radius=1, kernelType='square', units='pixels')

    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
           .select(['VV', 'VH']).map(clean))
    img = col.median()
    img = img.addBands(img.select('VV').subtract(img.select('VH')).rename('VV_VH'))
    return img, col.size()


def dynamic_world_month(region, month):
    """Most frequent Dynamic World label in the month, remapped to our 7 classes."""
    start = ee.Date(month + '-01')
    col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
           .filterBounds(region).filterDate(start, start.advance(1, 'month')).select('label'))
    return col.mode().remap(DW_FROM, DW_TO).rename('dw'), col.size()


def make_grid(lat, lon, size_km, scale_m, crs):
    """Pixel grid centred on (lat, lon) in an equal-area projection, so every pixel has the same area."""
    x, y = ee.Geometry.Point([lon, lat]).transform(crs, 1).coordinates().getInfo()
    n = int(size_km * 1000 / scale_m)
    half = n * scale_m / 2
    grid = {'dimensions': {'width': n, 'height': n},
            'affineTransform': {'scaleX': scale_m, 'shearX': 0, 'translateX': x - half,
                                'shearY': 0, 'scaleY': -scale_m, 'translateY': y + half},
            'crsCode': crs}
    region = ee.Geometry.Point([lon, lat]).buffer(half * 1.5).bounds()   # generous search area
    return grid, region


def download(image, bands, grid):
    """Download an ee.Image on the given grid as a (bands, H, W) float32 numpy array."""
    arr = ee.data.computePixels({'expression': image.select(bands), 'fileFormat': 'NUMPY_NDARRAY', 'grid': grid})
    return np.stack([arr[b] for b in bands]).astype(np.float32)


# ---------------------------------------------------------------- 2. inference
def load_model(weights, config, device):
    model = build_model(config['variant'], len(config['class_names']), pretrained_encoder=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, x_db, config, device):
    """x_db: (3, H, W) in dB. Returns (H, W) class map with IGNORE where there was no valid SAR data."""
    invalid = (x_db[0] <= -49) | np.isnan(x_db).any(0)
    mean = np.array(config['mean'], dtype=np.float32)[:, None, None]
    std = np.array(config['std'], dtype=np.float32)[:, None, None]
    x = (x_db - mean) / std                      # same normalisation as in training
    x[:, invalid] = 0.0
    H, W = invalid.shape
    pad_h, pad_w = (-H) % 32, (-W) % 32           # U-Net needs sizes divisible by 2^5
    t = torch.from_numpy(x)[None].to(device)
    t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
    pred = model(t).argmax(1)[0, :H, :W].cpu().numpy().astype(np.uint8)
    pred[invalid] = IGNORE
    return pred


# ---------------------------------------------------------------- 3. calculation
def class_percentages(maps, months, class_names):
    """% of the valid area covered by each class, per month (rows = months)."""
    rows = []
    for m, mp in zip(months, maps):
        if mp is None:
            rows.append([np.nan] * len(class_names))
            continue
        valid = mp[mp != IGNORE]
        if valid.size == 0:                      # e.g. a fully clouded Dynamic World month
            rows.append([np.nan] * len(class_names))
            continue
        counts = np.bincount(valid, minlength=len(class_names))[:len(class_names)]
        rows.append(100 * counts / valid.size)
    return pd.DataFrame(rows, index=months, columns=class_names)


def changes(pct):
    """Month-to-month change and total change (last month - first month), in percentage points."""
    mom = pct.diff().iloc[1:]
    mom.index = [f'{a} -> {b}' for a, b in zip(pct.index[:-1], pct.index[1:])]
    valid = pct.dropna()
    total = (valid.iloc[-1] - valid.iloc[0]).rename(f'{valid.index[0]} -> {valid.index[-1]}')
    return mom, total


def agreement(pred, ref, n_classes):
    """Overall agreement and mIoU between our map and a reference map (pixels valid in both)."""
    m = (pred != IGNORE) & (ref != IGNORE)
    if m.sum() == 0:
        return np.nan, np.nan
    cm = np.bincount(ref[m].astype(int) * n_classes + pred[m], minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        iou = tp / union
    return tp.sum() / cm.sum(), np.nanmean(iou[present])


# ---------------------------------------------------------------- 4. visualisation
def plot_maps(maps, months, config, title, path):
    cmap = ListedColormap(config['class_colors'])
    n = len(config['class_names'])
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    for ax, m, mp in zip(axes.ravel(), months, maps):
        if mp is None:
            ax.text(0.5, 0.5, 'no Sentinel-1 data', ha='center', va='center')
        else:
            ax.imshow(np.ma.masked_equal(mp, IGNORE), cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation='nearest')
        ax.set_title(m)
        ax.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in config['class_colors']]
    fig.legend(handles, config['class_names'], loc='lower center', ncol=n, fontsize=11)
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig(path, dpi=150)
    plt.show()


def plot_percentages(pct, config, title, path):
    ax = pct.plot(figsize=(12, 5), marker='o', color=config['class_colors'])
    ax.set(ylabel='% of area', xlabel='month', title=title)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


# ---------------------------------------------------------------- main pipeline
def run(lat, lon, start, project, weights='model_best.pt', config_path='model_config.json',
        out='results', size_km=10, compare_dynamic_world=True, authenticated=False):
    if not authenticated:
        ee.Initialize(project=project)
    os.makedirs(out, exist_ok=True)
    config = json.load(open(config_path))
    names = config['class_names']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(weights, config, device)
    months = month_list(start)
    grid, region = make_grid(lat, lon, size_km, config['scale_m'], config['crs'])
    print(f'Location ({lat}, {lon}), {size_km} x {size_km} km, months {months[0]} .. {months[-1]}, device {device}')

    # 1 + 2: retrieval, compositing and prediction for each month
    maps, dw_maps, log = [], [], []
    for m in months:
        img, n_scenes = s1_monthly_composite(region, m)
        n_scenes = n_scenes.getInfo()
        if n_scenes == 0:
            print(f'{m}: no Sentinel-1 scenes -> skipped')
            maps.append(None); dw_maps.append(None)
            log.append(dict(month=m, s1_scenes=0))
            continue
        pred = predict(model, download(img.unmask(-50), config['band_order'], grid), config, device)
        maps.append(pred)
        entry = dict(month=m, s1_scenes=n_scenes, valid_pct=round(100 * (pred != IGNORE).mean(), 1))
        if compare_dynamic_world:
            dw, n_dw = dynamic_world_month(region, m)
            n_dw = n_dw.getInfo()
            dw_map = download(dw.unmask(IGNORE), ['dw'], grid)[0].astype(np.uint8) if n_dw > 0 else None
            dw_maps.append(dw_map)
            entry['dw_images'] = n_dw
            if dw_map is not None:
                oa, miou = agreement(pred, dw_map, len(names))
                entry.update(agreement_OA=round(oa, 3), agreement_mIoU=round(miou, 3))
        log.append(entry)
        print(entry)

    # 3: calculation
    pct = class_percentages(maps, months, names)
    mom, total = changes(pct)
    pct.round(2).to_csv(f'{out}/class_percentages.csv')
    mom.round(2).to_csv(f'{out}/change_month_to_month_pp.csv')
    total.round(2).to_csv(f'{out}/change_total_pp.csv')
    pd.DataFrame(log).to_csv(f'{out}/run_log.csv', index=False)
    np.savez_compressed(f'{out}/lulc_maps.npz', **{m: mp for m, mp in zip(months, maps) if mp is not None})

    print('\n% of area per class:'); print(pct.round(2).to_string())
    print('\nMonth-to-month change (percentage points):'); print(mom.round(2).to_string())
    print(f'\nTotal change {total.name} (percentage points):'); print(total.round(2).to_string())

    # 4: visualisation
    title = f'Sentinel-1 LULC ({lat:.4f}, {lon:.4f}), {months[0]} to {months[-1]}'
    plot_maps(maps, months, config, title, f'{out}/lulc_maps_12_months.png')
    plot_percentages(pct, config, 'Class percentages per month (model)', f'{out}/class_percentages.png')

    # 5: independent sanity check against Dynamic World
    if compare_dynamic_world and any(d is not None for d in dw_maps):
        dw_pct = class_percentages(dw_maps, months, names)
        dw_pct.round(2).to_csv(f'{out}/dynamic_world_percentages.csv')
        _, dw_total = changes(dw_pct)
        summary = pd.DataFrame({'model mean %': pct.mean(), 'Dynamic World mean %': dw_pct.mean(),
                                'model total change pp': total, 'Dynamic World total change pp': dw_total}).round(2)
        summary.to_csv(f'{out}/comparison_dynamic_world.csv')
        print('\nComparison with Dynamic World (independent, Sentinel-2 based):'); print(summary.to_string())
        plot_maps(dw_maps, months, config, 'Dynamic World reference (remapped to 7 classes)',
                  f'{out}/dynamic_world_12_months.png')
    print(f'\nAll results saved in: {out}')
    return dict(maps=maps, percentages=pct, month_to_month=mom, total=total, log=pd.DataFrame(log))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Sentinel-1 LULC prototype')
    p.add_argument('--lat', type=float, required=True)
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--start', required=True, help='start month, format YYYY-MM')
    p.add_argument('--project', required=True, help='Google Cloud project registered for Earth Engine')
    p.add_argument('--weights', default='model_best.pt')
    p.add_argument('--config', default='model_config.json')
    p.add_argument('--out', default='results')
    p.add_argument('--size_km', type=float, default=10)
    p.add_argument('--no_dynamic_world', action='store_true')
    a = p.parse_args()
    run(a.lat, a.lon, a.start, a.project, a.weights, a.config, a.out, a.size_km, not a.no_dynamic_world)
