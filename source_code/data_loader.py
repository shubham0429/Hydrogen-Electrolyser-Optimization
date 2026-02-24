# src/load_and_preview_mat.py
from pathlib import Path
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def inspect_mat(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    print("Top-level keys in .mat:", [k for k in mat.keys() if not k.startswith('__')])
    return mat

def try_extract_timeseries(mat, pv_key=None, wind_key=None, time_key=None):
    # If you know the variable names, pass them. Otherwise we'll try common guesses.
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print("Available keys:", keys)

    # Try to auto-find likely time and data arrays
    if time_key is None:
        for guess in ("time","t","timestamp","time_s","datetime"):
            if guess in mat: 
                time_key = guess; break
    if pv_key is None:
        for guess in ("pv","PV","pv_power","pv_kw","pv_mw"):
            if guess in mat:
                pv_key = guess; break
    if wind_key is None:
        for guess in ("wind","Wind","wind_power","wind_kw","wind_mw"):
            if guess in mat:
                wind_key = guess; break

    print("Using time_key, pv_key, wind_key:", time_key, pv_key, wind_key)
    df = pd.DataFrame()

    # convert time if present
    if time_key and time_key in mat:
        t = np.array(mat[time_key]).squeeze()
        # If MATLAB datenum (large numbers), convert:
        if t.dtype.kind in ('f','i') and t.mean() > 70000: # likely datenum
            # Matlab datenum to pandas datetime: datenum->ordinal + offset
            df['time'] = pd.to_datetime(t - 719529, unit='D')  # adjust if needed
        else:
            try:
                df['time'] = pd.to_datetime(t, unit='s')  # if epoch seconds
            except Exception:
                df['time'] = pd.to_datetime(t)  # fallback
    else:
        # create a simple index 0..N-1
        length = None
        for k in (pv_key, wind_key):
            if k and k in mat:
                length = np.array(mat[k]).squeeze().shape[0]
                break
        df['time'] = pd.date_range(start="2020-01-01", periods=length, freq='H')

    # add PV and wind if found
    if pv_key and pv_key in mat:
        df['pv'] = np.array(mat[pv_key]).squeeze()
    if wind_key and wind_key in mat:
        df['wind'] = np.array(mat[wind_key]).squeeze()

    # if shape is (n,1) or (1,n), squeeze to 1D
    df = df.set_index('time')
    print("Preview of dataframe:")
    print(df.head())
    return df

def save_and_plot(df, outdir="results"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    csv = Path(outdir)/"power_timeseries_preview.csv"
    df.to_csv(csv)
    print("Saved CSV:", csv)

    # simple plots
    plt.figure(figsize=(8,3))
    if 'pv' in df: plt.plot(df.index, df['pv'], label='PV')
    if 'wind' in df: plt.plot(df.index, df['wind'], label='Wind')
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(outdir)/"power_timeseries.png", dpi=150)
    plt.close()
    print("Saved plot:", Path(outdir)/"power_timeseries.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/load_and_preview_mat.py path/to/data.mat")
        sys.exit(1)
    path = sys.argv[1]
    mat = inspect_mat(path)
    df = try_extract_timeseries(mat)  # pass names if you know them
    save_and_plot(df)
