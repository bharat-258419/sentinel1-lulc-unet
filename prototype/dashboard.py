"""LULC dashboard: interactive web interface for the Sentinel-1 LULC prototype.

Start with:   streamlit run dashboard.py
The dashboard uses the same functions as lulc_prototype.py (retrieval, compositing, prediction, calculation).
"""
import json
import os

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import ListedColormap

import lulc_prototype as lp

PROJECT = os.environ.get('EE_PROJECT', 'lulc-bharat-2026')   # your Earth Engine Cloud project
CITIES = {
    'Mannheim (case study)': (49.4875, 8.4660),
    'Heidelberg': (49.4093, 8.6944),
    'Karlsruhe': (49.0069, 8.4037),
    'Frankfurt am Main': (50.1109, 8.6821),
    'Berlin': (52.5200, 13.4050),
    'Custom location': None,
}

st.set_page_config(page_title='Sentinel-1 LULC Dashboard', layout='wide')


# ---------------------------------------------------------------- cached resources
@st.cache_resource
def load_everything():
    ee.Initialize(project=PROJECT)
    config = json.load(open('model_config.json'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = lp.load_model('model_best.pt', config, device)
    return config, model, device


config, model, device = load_everything()
CLASSES = config['class_names']
COLORS = config['class_colors']
CMAP = ListedColormap(COLORS)


# ---------------------------------------------------------------- figures
def map_grid(maps, months, title):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
    for ax, m, mp in zip(axes.ravel(), months, maps):
        if mp is None or (mp != lp.IGNORE).sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=14, color='grey')
        else:
            ax.imshow(np.ma.masked_equal(mp, lp.IGNORE), cmap=CMAP, vmin=-0.5, vmax=len(CLASSES) - 0.5,
                      interpolation='nearest')
        ax.set_title(m, fontsize=13)
        ax.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, CLASSES, loc='lower center', ncol=len(CLASSES), fontsize=12)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def single_map(mp, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.ma.masked_equal(mp, lp.IGNORE), cmap=CMAP, vmin=-0.5, vmax=len(CLASSES) - 0.5,
              interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


def percentage_chart(pct):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, color in zip(CLASSES, COLORS):
        ax.plot(pct.index, pct[name], marker='o', color=color, label=name)
    ax.set(ylabel='% of area', xlabel='month')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    return fig


def total_change_chart(total):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(total.index, total.values, color=COLORS)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('change (percentage points)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar: input
st.sidebar.title('Input')
city = st.sidebar.selectbox('Location', list(CITIES))
if CITIES[city] is None:
    lat = st.sidebar.number_input('Latitude', value=49.4875, format='%.4f')
    lon = st.sidebar.number_input('Longitude', value=8.4660, format='%.4f')
else:
    lat, lon = CITIES[city]
    st.sidebar.write(f'Latitude **{lat:.4f}**, longitude **{lon:.4f}**')

start_options = [str(p) for p in pd.period_range('2017-01', '2025-12', freq='M')]
start = st.sidebar.selectbox('Start month', start_options, index=start_options.index('2025-01'))
size_km = st.sidebar.slider('Area size (km)', 4, 12, 10)
compare_dw = st.sidebar.checkbox('Compare with Dynamic World', value=True)
run = st.sidebar.button('Run prototype', type='primary', use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.caption(f"Model: U-Net ({config['variant']}), trained on Sentinel-1 VV, VH, VV−VH. "
                   'Labels: ESA WorldCover 2021. Device: ' + device)

# ---------------------------------------------------------------- header
st.title('Sentinel-1 Land Use / Land Cover Dashboard')
st.write('Monthly land use and land cover maps from **Sentinel-1 radar data only**, '
         'produced by a U-Net semantic segmentation model.')

# ---------------------------------------------------------------- processing
if run:
    months = lp.month_list(start)
    grid, region = lp.make_grid(lat, lon, size_km, config['scale_m'], config['crs'])
    maps, dw_maps, log = [], [], []
    progress = st.progress(0.0, text='Starting ...')
    for i, m in enumerate(months):
        progress.progress(i / 12, text=f'Processing {m}: Sentinel-1 retrieval, monthly composite, prediction ...')
        img, n = lp.s1_monthly_composite(region, m)
        n = n.getInfo()
        if n == 0:
            maps.append(None); dw_maps.append(None)
            log.append(dict(month=m, s1_scenes=0))
            continue
        pred = lp.predict(model, lp.download(img.unmask(-50), config['band_order'], grid), config, device)
        maps.append(pred)
        entry = dict(month=m, s1_scenes=n)
        if compare_dw:
            dw, n_dw = lp.dynamic_world_month(region, m)
            n_dw = n_dw.getInfo()
            dw_map = lp.download(dw.unmask(lp.IGNORE), ['dw'], grid)[0].astype(np.uint8) if n_dw > 0 else None
            dw_maps.append(dw_map)
            oa, miou = lp.agreement(pred, dw_map, len(CLASSES)) if dw_map is not None else (np.nan, np.nan)
            entry.update(dw_images=n_dw, agreement_OA=oa)
        log.append(entry)
    progress.progress(1.0, text='Done')
    st.session_state['result'] = dict(lat=lat, lon=lon, months=months, maps=maps, dw_maps=dw_maps,
                                      log=pd.DataFrame(log), compare_dw=compare_dw, size_km=size_km)

# ---------------------------------------------------------------- results
res = st.session_state.get('result')
if res is None:
    st.info('Choose a location and a start month in the sidebar and click **Run prototype**.')
    st.stop()

months, maps = res['months'], res['maps']
pct = lp.class_percentages(maps, months, CLASSES)
mom, total = lp.changes(pct)
valid = pct.dropna()
last = valid.index[-1]

st.subheader(f"Results for ({res['lat']:.4f}, {res['lon']:.4f}), {res['size_km']} × {res['size_km']} km, "
             f'{months[0]} to {months[-1]}')

# key figures for the last month
cols = st.columns(len(CLASSES))
for col, name in zip(cols, CLASSES):
    col.metric(name, f'{valid.loc[last, name]:.1f} %', f'{total[name]:+.2f} pp', delta_color='off')
st.caption(f'Share of each class in {last}; the small number is the total change since {valid.index[0]} '
           '(percentage points).')

tab_maps, tab_month, tab_pct, tab_change, tab_dw = st.tabs(
    ['12 monthly maps', 'Single month', 'Class percentages', 'Changes', 'Check vs. Dynamic World'])

with tab_maps:
    st.pyplot(map_grid(maps, months, 'Sentinel-1 LULC maps'))

with tab_month:
    available = [m for m, mp in zip(months, maps) if mp is not None]
    m_sel = st.select_slider('Month', options=available, value=available[0])
    mp = maps[months.index(m_sel)]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(single_map(mp, f'Model prediction {m_sel}'))
    with c2:
        if res['compare_dw']:
            dmp = res['dw_maps'][months.index(m_sel)]
            if dmp is not None and (dmp != lp.IGNORE).any():
                st.pyplot(single_map(dmp, f'Dynamic World {m_sel}'))
            else:
                st.warning('No Dynamic World data for this month (cloud cover in the optical images).')
    st.dataframe(pct.loc[[m_sel]].round(2), use_container_width=True)

with tab_pct:
    st.pyplot(percentage_chart(pct))
    st.dataframe(pct.round(2), use_container_width=True)

with tab_change:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f'**Total change {valid.index[0]} → {last}** (percentage points)')
        st.pyplot(total_change_chart(total))
    with c2:
        st.markdown('**Month-to-month change** (percentage points)')
        st.dataframe(mom.round(2), use_container_width=True)

with tab_dw:
    if not res['compare_dw']:
        st.info('Dynamic World comparison was switched off.')
    else:
        dw_pct = lp.class_percentages(res['dw_maps'], months, CLASSES)
        _, dw_total = lp.changes(dw_pct)
        log = res['log']
        st.markdown('Google Dynamic World is an independent LULC map from **Sentinel-2 optical** data. '
                    'It was **not used for training**, so it serves as a sanity check.')
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('**Pixel agreement per month**')
            st.dataframe(log.round(3), use_container_width=True, hide_index=True)
            st.metric('Mean agreement (OA)', f"{np.nanmean(log['agreement_OA']) * 100:.1f} %")
        with c2:
            st.markdown('**Class shares and total change**')
            comp = pd.DataFrame({'model mean %': pct.mean(), 'Dynamic World mean %': dw_pct.mean(),
                                 'model change pp': total, 'Dynamic World change pp': dw_total}).round(2)
            st.dataframe(comp, use_container_width=True)
        st.pyplot(map_grid(res['dw_maps'], months, 'Dynamic World reference (remapped to 7 classes)'))
