"""
imu50_fetch.py — daytime wrist segments of the IMU50 dataset (healthy people, 5 days, free living).

Dataset: IMU50, Zenodo record 21468410 (CC BY 4.0). 50 healthy volunteers aged 40-75 (no motor,
neurological, cardiovascular or respiratory disease), ActiGraph LEAP on the non-dominant wrist for 5 days
and nights; acc (g) + gyro (deg/s) at 128 Hz. One 46.7 GB zip that holds one stored (uncompressed) zip per
subject, so a subject's IMU csv is streamed with HTTP range requests; nothing is saved to disk except the
windows kept.

Per subject: for every hour from HOUR_FROM to HOUR_TO (local time), the first SEG_MIN minutes = one segment
(matches exp6, where a segment is the minutes before a symptom report). Each segment: gyro -> rad/s,
resampled 128 -> 100 Hz, band-passed 1-20 Hz, 2-s windows; still windows (acc-magnitude SD < STILL_G)
dropped; at most MAX_WIN windows kept, evenly spaced.

Writes IMU50_DIR/<subject>.npz (windows float16 (n, 6, 200), segment id, intensity) and segments.csv
(subject, segment, day, hour, n windows). Resumable. Streaming is CPU-bound (csv parsing, ~1-3 min per subject);
reads are 64 MB ranges (~12 per subject) to stay under Zenodo's request rate limit.

    python classifier_vs_scan/imu50_fetch.py
    SUBJECTS=05 MAX_ROWS=3000000 IMU50_DIR=/tmp/i python classifier_vs_scan/imu50_fetch.py   # quick test
"""

import os, os.path as osp, sys, io, time, zipfile
from concurrent.futures import ProcessPoolExecutor
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

URL       = 'https://zenodo.org/api/records/21468410/files/IMU50.zip/content'
OUT       = os.environ.get('IMU50_DIR', osp.join(C.RESULTS_DIR, 'classifier_vs_scan', 'imu50'))
SEG_MIN   = float(os.environ.get('SEG_MIN', '10'))
HOUR_FROM, HOUR_TO = int(os.environ.get('HOUR_FROM', '8')), int(os.environ.get('HOUR_TO', '20'))
STILL_G   = float(os.environ.get('STILL_G', '0.01'))
MAX_WIN   = int(os.environ.get('MAX_WIN', '60'))
SUBJECTS  = [s for s in os.environ.get('SUBJECTS', '').split(',') if s]
MAX_ROWS  = int(os.environ.get('MAX_ROWS', '0'))
WORKERS   = int(os.environ.get('WORKERS', '4'))
SRC_FS    = 128


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""
    def __init__(self, url):
        self.url, self.ses, self.pos = url, requests.Session(), 0
        r = self._get(0, 0)
        self.size = int(r.headers['Content-Range'].split('/')[1])

    def _get(self, a, b):
        # Zenodo allows ~133 requests/min per IP: honour 429 Retry-After, back off on errors
        for i in range(10):
            try:
                r = self.ses.get(self.url, headers={'Range': f'bytes={a}-{b}'}, timeout=300)
                if r.status_code == 206:
                    return r
                wait = float(r.headers.get('Retry-After', 2 ** min(i, 6)))
            except requests.RequestException:
                wait = 2 ** min(i, 6)
            time.sleep(wait)
        raise RuntimeError(f'range read failed at {a}')

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, o, whence=0):
        self.pos = o if whence == 0 else self.pos + o if whence == 1 else self.size + o
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        d = self._get(self.pos, self.pos + n - 1).content
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def open_outer():
    return zipfile.ZipFile(io.BufferedReader(HTTPFile(URL), buffer_size=64 << 20))


def segment_windows_of(x):
    """x: (T, 6) at 128 Hz, acc g + gyro deg/s -> filtered windows (n, 6, 200), intensity (n,)."""
    sig = x.T.astype(np.float32)
    sig[3:6] *= np.pi / 180
    sig = resample_poly(sig, 25, 32, axis=1).astype(np.float32)          # 128 -> 100 Hz
    if sig.shape[1] < C.WINDOW_SIZE:
        return None, None
    raw_w = segment_windows(sig, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)
    intensity = np.linalg.norm(raw_w[:, 0:3], axis=1).std(axis=1)
    w = segment_windows(bandpass_filter(sig, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS),
                        window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)
    keep = np.where(intensity >= STILL_G)[0]
    if len(keep) > MAX_WIN:
        keep = keep[np.linspace(0, len(keep) - 1, MAX_WIN).round().astype(int)]
    return w[keep], intensity[keep]


def do_subject(sid):
    t0 = time.time()
    path = osp.join(OUT, f'{sid}.npz')
    outer = open_outer()
    inner = zipfile.ZipFile(outer.open(f'IMU50/DATA/{sid}.zip'))
    seg_len = f'{int(SEG_MIN):02d}'
    W, SG, I, rows = [], [], [], []
    buf, cur = [], None
    first_day = None
    with inner.open(f'{sid}/{sid}_imu.csv') as fh:
        reader = pd.read_csv(fh, chunksize=2_000_000, dtype={c: np.float32 for c in range(1, 7)},
                             nrows=MAX_ROWS or None)
        for chunk in reader:
            ts = chunk['Timestamp'].astype(str)
            hour = ts.str[11:13].astype(int).values
            minute = ts.str[14:16].values
            sel = (hour >= HOUR_FROM) & (hour <= HOUR_TO) & (minute < seg_len)
            if not sel.any():
                continue
            key = ts.str[:13].values                                     # 'YYYY-MM-DD HH'
            vals = chunk.iloc[:, 1:7].values
            for k in pd.unique(key[sel]):
                m = sel & (key == k)
                if cur is not None and k != cur:
                    rows.append(flush(cur, buf, W, SG, I, len(rows), first_day))
                    buf = []
                if first_day is None:
                    first_day = pd.Timestamp(k[:10])
                cur = k
                buf.append(vals[m])
    if buf:
        rows.append(flush(cur, buf, W, SG, I, len(rows), first_day))
    rows = [r for r in rows if r is not None]
    if not W:
        return sid, None, time.time() - t0
    np.savez_compressed(path, windows=np.concatenate(W), segment=np.concatenate(SG).astype(np.int32),
                        intensity=np.concatenate(I).astype(np.float32))
    return sid, pd.DataFrame(rows), time.time() - t0


def flush(key, buf, W, SG, I, seg_id, first_day):
    x = np.concatenate(buf)
    w, inten = segment_windows_of(x)
    n = 0 if w is None else len(w)
    if n:
        W.append(w.astype(np.float16)); SG.append(np.full(n, seg_id)); I.append(inten)
    return {'segment': seg_id, 'hour_key': key, 'day': (pd.Timestamp(key[:10]) - first_day).days,
            'hour': int(key[11:13]), 'n_samples': len(x), 'n_windows': n}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    outer = open_outer()
    names = [n for n in outer.namelist() if n.startswith('IMU50/DATA/') and n.endswith('.zip')]
    sids = SUBJECTS or sorted(osp.basename(n)[:-4] for n in names)
    with open(osp.join(OUT, 'subjects_info.csv'), 'wb') as f:
        f.write(outer.read('IMU50/DATA/subjects_info.csv'))
    seg_path = osp.join(OUT, 'segments.csv')
    done = pd.read_csv(seg_path, dtype={'subject': str}) if osp.exists(seg_path) else pd.DataFrame()
    todo = [s for s in sids if not (osp.exists(osp.join(OUT, f'{s}.npz')) and len(done) and (done['subject'] == s).any())]
    print(f'{len(sids)} subjects, {len(todo)} to fetch, segments {HOUR_FROM}:00-{HOUR_TO}:{int(SEG_MIN):02d} -> {OUT}')
    tables = [done] if len(done) else []
    with ProcessPoolExecutor(WORKERS) as ex:
        for sid, tab, sec in ex.map(do_subject, todo):
            if tab is None:
                print(f'  {sid}: no windows ({sec:.0f} s)'); continue
            tab['subject'] = sid
            tables.append(tab)
            pd.concat(tables, ignore_index=True).to_csv(seg_path, index=False)
            print(f"  {sid}: {len(tab)} segments over {tab['day'].nunique()} days, {tab['n_windows'].sum()} windows ({sec:.0f} s)")
    print(f'DONE {time.time() - t_start:.0f} s')
