"""
habets_fetch.py — download the wrist-sensor minutes around every symptom report of the Habets et al.
home-monitoring dataset and window them exactly like PADS.

Dataset: Habets & Kubben, "EMA and wearable sensor monitoring in PD", DataverseNL,
doi:10.34894/5HHK8H (CC0). 20 people with PD, ~14 days at home, sensors on both wrists and the chest
(200 Hz, acc in g, gyro in deg/s, EDF), plus ~7 smartphone symptom questionnaires (EMA "beeps") a day.
Data descriptor: Habets et al., Data 2021, 6(2):22.

The full sensor set is 107 GB. Only the PRE_MIN minutes before each beep are needed, and EDF records
have a fixed size (1 s x 6 channels x 200 samples x int16 = 2400 bytes), so each segment is read with
an HTTP byte-range request (~1.4 MB per beep per wrist at PRE_MIN=10; ~4-5 GB in total).

Per beep and wrist: the segment is converted to physical units (gyro -> rad/s unless GYRO_TO_RAD=0),
resampled 200 -> 100 Hz, band-passed 1-20 Hz and cut into 2-s windows (same as PADS). Windows where the
wrist is still (raw acc magnitude SD < STILL_G) are dropped (non-wear / no movement); at most MAX_WIN
windows per beep and wrist are kept, evenly spaced.

Writes HABETS_DIR/<patient>.npz (windows float16 (n, 6, 200), beep index, side, intensity) and
HABETS_DIR/beeps.csv (EMA answers + per-wrist coverage). Resumable: finished patients are skipped.

    python classifier_vs_scan/habets_fetch.py
    PATIENTS=110001 MAX_BEEPS=5 HABETS_DIR=/tmp/h python classifier_vs_scan/habets_fetch.py   # quick test
"""

import os, os.path as osp, sys, io, json, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from scipy.signal import resample_poly

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, 'scripts'))
import config as C                                                       # noqa: E402
from src.data.preprocessing import bandpass_filter, segment_windows     # noqa: E402

DOI       = 'doi:10.34894/5HHK8H'
API       = 'https://dataverse.nl/api'
OUT       = os.environ.get('HABETS_DIR', osp.join(C.RESULTS_DIR, 'classifier_vs_scan', 'habets'))
PRE_MIN   = float(os.environ.get('PRE_MIN', '10'))      # minutes before the beep
MIN_COVER = 0.5                                         # keep a segment if >= 50% of it was recorded
STILL_G   = float(os.environ.get('STILL_G', '0.01'))    # raw acc-magnitude SD below this = still
MAX_WIN   = int(os.environ.get('MAX_WIN', '60'))
GYRO_TO_RAD = os.environ.get('GYRO_TO_RAD', '1') == '1'
PATIENTS  = [p for p in os.environ.get('PATIENTS', '').split(',') if p]
MAX_BEEPS = int(os.environ.get('MAX_BEEPS', '0'))       # >0: only the first n beeps per patient (testing)
THREADS   = int(os.environ.get('THREADS', '8'))
SENSORS   = {'left': {'13797', '13799', '13794', '13806'}, 'right': {'13805', '13801', '13793', '13795'}}
REC_BYTES = 6 * 200 * 2
SRC_FS    = 200

MOTOR_ITEMS = ['tremor', 'slowness', 'stiffness', 'walk_well', 'mobility_well', 'dyskinesia', 'sanpar_onoff']

session = requests.Session()


def get(url, rng=None, tries=5):
    for i in range(tries):
        try:
            r = session.get(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}'} if rng else {}, timeout=120)
            if r.status_code in (200, 206):
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f'failed: {url} {rng}')


def file_table():
    meta = json.loads(get(f'{API}/datasets/:persistentId/?persistentId={DOI}'))['data']['latestVersion']
    rows, ema_id = [], None
    for f in meta['files']:
        df = f['dataFile']
        name = df['filename']
        if name == 'EMA_data.csv':
            ema_id = df['id']
        if not name.endswith('.edf'):
            continue
        sensor, day, tod = name[:-4].split('_')
        side = next((s for s, ids in SENSORS.items() if sensor in ids), None)
        if side is None:
            continue                                            # chest sensor
        rows.append({'patient': str(f.get('directoryLabel', '')), 'side': side, 'id': df['id'], 'name': name,
                     'start': dt.datetime.strptime(day + tod, '%Y%m%d%H%M%S'), 'size': df['filesize']})
    return pd.DataFrame(rows), ema_id


def edf_header(fid):
    h = get(f'{API}/access/datafile/{fid}', (0, 255)).decode('latin1')
    ns, hb = int(h[252:256]), int(h[184:192])
    s = get(f'{API}/access/datafile/{fid}', (256, hb - 1)).decode('latin1')
    fld = lambda off, w: [s[off + i * w: off + (i + 1) * w].strip() for i in range(ns)]  # noqa: E731
    names = fld(0, 16)
    o = ns * (16 + 80 + 8)
    pmin = np.array(fld(o, 8), float); o += 8 * ns
    pmax = np.array(fld(o, 8), float); o += 8 * ns
    dmin = np.array(fld(o, 8), float); o += 8 * ns
    dmax = np.array(fld(o, 8), float); o += 8 * ns + 80 * ns
    nsr = np.array(fld(o, 8), int)
    assert ns == 6 and (nsr == SRC_FS).all(), (names, nsr)
    gain = (pmax - pmin) / (dmax - dmin)
    return {'hb': hb, 'names': names, 'gain': gain, 'offset': pmin - dmin * gain}


def read_segment(row, hdr, t0, t1):
    """Physical signal (6, T) at 200 Hz for [t0, t1) clipped to the file; None if too short."""
    n_rec = (row['size'] - hdr['hb']) // REC_BYTES
    a = max(0, int(np.floor((t0 - row['start']).total_seconds())))
    b = min(n_rec, int(np.ceil((t1 - row['start']).total_seconds())))
    if b <= a:
        return None, 0
    raw = get(f"{API}/access/datafile/{row['id']}", (hdr['hb'] + a * REC_BYTES, hdr['hb'] + b * REC_BYTES - 1))
    x = np.frombuffer(raw, '<i2').reshape(b - a, 6, SRC_FS).transpose(1, 0, 2).reshape(6, -1).astype(np.float32)
    x = x * hdr['gain'][:, None] + hdr['offset'][:, None]
    if GYRO_TO_RAD:
        x[3:6] *= np.pi / 180
    return x, b - a


def windows_of(sig200):
    sig = resample_poly(sig200, 1, 2, axis=1).astype(np.float32)          # 200 -> 100 Hz
    if sig.shape[1] < C.WINDOW_SIZE:
        return None, None
    raw_w = segment_windows(sig, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)
    intensity = np.linalg.norm(raw_w[:, 0:3], axis=1).std(axis=1)          # acc magnitude SD per window (g)
    w = segment_windows(bandpass_filter(sig, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS),
                        window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)
    keep = np.where(intensity >= STILL_G)[0]
    if len(keep) > MAX_WIN:
        keep = keep[np.linspace(0, len(keep) - 1, MAX_WIN).round().astype(int)]
    return w[keep], intensity[keep]


def do_patient(pid, files, ema, headers):
    path = osp.join(OUT, f'{pid}.npz')
    pe = ema[ema['ID'].astype(str) == pid].reset_index(drop=True)
    if MAX_BEEPS:
        pe = pe.iloc[:MAX_BEEPS]
    cover = {'left': [], 'right': []}
    W, B, SD, I = [], [], [], []

    def job(args):
        bi, side, t = args
        t0, t1 = t - dt.timedelta(minutes=PRE_MIN), t
        cand = files[(files['side'] == side) & (files['start'] < t1)]
        best, got = None, 0
        for _, row in cand.iterrows():
            n_rec = (row['size'] - headers[row['id']]['hb']) // REC_BYTES
            end = row['start'] + dt.timedelta(seconds=int(n_rec))
            ov = (min(end, t1) - max(row['start'], t0)).total_seconds()
            if ov > got:
                best, got = row, ov
        if best is None or got < MIN_COVER * PRE_MIN * 60:
            return bi, side, None, None, 0.0
        sig, n = read_segment(best, headers[best['id']], t0, t1)
        w, inten = windows_of(sig)
        return bi, side, w, inten, n / (PRE_MIN * 60)

    jobs = [(bi, side, t) for bi, t in enumerate(pe['beep_time_start']) for side in ('left', 'right')]
    with ThreadPoolExecutor(THREADS) as ex:
        for bi, side, w, inten, cov in ex.map(job, jobs):
            cover[side].append((bi, round(cov, 3), 0 if w is None else len(w)))
            if w is not None and len(w):
                W.append(w.astype(np.float16)); B += [bi] * len(w); SD += [side] * len(w); I.append(inten)
    if not W:
        print(f'  {pid}: no windows'); return None
    np.savez_compressed(path, windows=np.concatenate(W), beep=np.array(B, np.int32), side=np.array(SD),
                        intensity=np.concatenate(I).astype(np.float32))
    for side in ('left', 'right'):
        c = dict((b, (cv, n)) for b, cv, n in cover[side])
        pe[f'cover_{side}'] = [c.get(b, (0, 0))[0] for b in range(len(pe))]
        pe[f'nwin_{side}'] = [c.get(b, (0, 0))[1] for b in range(len(pe))]
    pe['beep'] = np.arange(len(pe))
    return pe


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    files, ema_id = file_table()
    ema = pd.read_csv(io.BytesIO(get(f'{API}/access/datafile/{ema_id}')), parse_dates=['beep_time_start', 'beep_time_end'])
    ema = ema.dropna(subset=['beep_time_start']).sort_values(['ID', 'beep_time_start'])
    pids = sorted(files['patient'].unique()) if not PATIENTS else PATIENTS
    print(f'{len(files)} wrist files, {len(pids)} patients, {len(ema)} EMA rows; PRE_MIN={PRE_MIN} -> {OUT}')

    beeps_path = osp.join(OUT, 'beeps.csv')
    done = pd.read_csv(beeps_path, dtype={'ID': str}) if osp.exists(beeps_path) else pd.DataFrame()
    tables = [done] if len(done) else []
    for pid in pids:
        if osp.exists(osp.join(OUT, f'{pid}.npz')) and len(done) and (done['ID'].astype(str) == pid).any():
            print(f'  {pid}: done'); continue
        pf = files[files['patient'] == pid].reset_index(drop=True)
        with ThreadPoolExecutor(THREADS) as ex:
            headers = dict(zip(pf['id'], ex.map(edf_header, pf['id'])))
        t0 = time.time()
        pe = do_patient(pid, pf, ema, headers)
        if pe is None:
            continue
        pe['ID'] = pid
        tables.append(pe)
        pd.concat(tables, ignore_index=True).to_csv(beeps_path, index=False)
        z = np.load(osp.join(OUT, f'{pid}.npz'))
        print(f"  {pid}: {len(pe)} beeps, {len(z['beep'])} windows, coverage L {pe['cover_left'].mean():.2f} "
              f"R {pe['cover_right'].mean():.2f} ({time.time() - t0:.0f} s)")
    print(f'DONE {time.time() - t_start:.0f} s')
