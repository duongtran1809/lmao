import pandas as pd
import numpy as np
import yfinance as yf
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew
import warnings
import sys
from datetime import datetime
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURATION & PORTFOLIO
# ==========================================
# List of (Ticker, SectorCodes)
# Example: [('CII.VN', 'REA, BNK'), ('FPT.VN', 'SMC'), ('VCB.VN', 'BNK')]
# BNK, STL, DLY, SFD, SMC, REA, ENG, FTL, RTL, SHP ('' for INTERNAL-ONLY)
PORTFOLIO_INPUT = [
    ('VCB.VN', 'BNK'),
    ('BID.VN', 'BNK'),
    ('CTG.VN', 'BNK'),
    ('TCB.VN', 'BNK'),
    ('MBB.VN', 'BNK'),
    ('VPB.VN', 'BNK'),
    ('ACB.VN', 'BNK'),
    ('STB.VN', 'BNK'),
    ('HDB.VN', 'BNK'),
    ('VIB.VN', 'BNK')
]

TYPE = 'Open'
START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
RANDOM_SEED = 42

# Strategy Parameters
HORIZON_BLOCKS = 10
POPSIZE = 10 # Optimized for portfolio batching
N_ITER_BO = 8
BATCH_SIZE_BO = 3

# External Feature Mapping
EXT_MAP = {
    'BNK': ['USDVND=X', '^TNX', '^IRX'],
    'STL': ['HRC=F', 'CL=F', 'BDRY'],
    'DLY': ['ZC=F', 'ZM=F', 'CL=F'],
    'SFD': ['USDVND=X', 'BDRY', 'CL=F'],
    'SMC': ['^IXIC', 'SOXX', 'HG=F', '^TWII', 'DX-Y.NYB'],
    'REA': ['USDVND=X', 'HRC=F', '^TNX'],
    'ENG': ['BZ=F', 'NG=F', 'DX-Y.NYB'],
    'FTL': ['NG=F', 'ZR=F', 'ZC=F', 'ZW=F'],
    'RTL': ['USDVND=X', 'CL=F', 'DX-Y.NYB', 'BDRY', 'TIP', 'XLY', 'VNM'],
    'SHP': ['BDRY', 'CL=F'],
}

np.random.seed(RANDOM_SEED)

# ==========================================
# 1. GP Core Engine
# ==========================================
def rbf_kernel(train, test, l=1.0):
    train, test = np.atleast_2d(train), np.atleast_2d(test)
    if train.shape[1] == 0: return None
    dist_sq = np.sum(train**2, axis=1).reshape(-1, 1) + np.sum(test**2, axis=1) - 2*(train @ test.T)
    return np.exp(-0.5 * np.maximum(dist_sq, 0) / (l ** 2))

def compute_dual_kernel(Ci1, Ci2, Ce1, Ce2, li, le, sigma, w):
    Ki = rbf_kernel(Ci1, Ci2, li)
    Ke = rbf_kernel(Ce1, Ce2, le)
    if Ki is not None and Ke is not None:
        return (sigma**2) * (w * Ki + (1 - w) * Ke)
    return (sigma**2) * (Ki if Ki is not None else Ke)

def gaussian_process_dual(Ci1, Ci2, Ce1, Ce2, li, le, sigma, sn, w, Y):
    K = compute_dual_kernel(Ci1, Ci1, Ce1, Ce1, li, le, sigma, w) + (sn**2) * np.eye(len(Ci1))
    Ks = compute_dual_kernel(Ci1, Ci2, Ce1, Ce2, li, le, sigma, w)
    Kss = compute_dual_kernel(Ci2, Ci2, Ce2, Ce2, li, le, sigma, w)
    L = cholesky(K + np.eye(len(K))*1e-6, lower=True)
    v = solve_triangular(L, Ks, lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, Y, lower=True), lower=False)
    mu = (Ks.T @ alpha).flatten()
    std = np.sqrt(np.maximum(np.diag(Kss - (v.T @ v)) + (sn**2), 0))
    return mu, std

def neg_log_likelihood_dual(params, Ci, Ce, w, y):
    li, le, sigma, sn = np.exp(params)
    K = compute_dual_kernel(Ci, Ci, Ce, Ce, li, le, sigma, w) + (sn**2) * np.eye(len(Ci)) + np.eye(len(Ci))*1e-6
    try: L = cholesky(K, lower=True)
    except: return np.inf
    alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)
    return 0.5*(y.T @ alpha).item() + np.sum(np.log(np.diag(L))) + (len(Ci)/2)*np.log(2*np.pi)

# ==========================================
# 2. Pipeline
# ==========================================
def process_single_stock(ticker, sector_codes):
    print(f"\nProcessing {ticker} (Sectors: {sector_codes})...")
    
    # Data Fetching
    stock_data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if stock_data.empty: return None
    if isinstance(stock_data.columns, pd.MultiIndex): stock_data.columns = stock_data.columns.get_level_values(0)
    
    # External features
    codes = [s.strip() for s in sector_codes.split(',') if s.strip()]
    ext_tickers = []
    for c in codes: ext_tickers.extend(EXT_MAP.get(c, []))
    ext_tickers = sorted(list(set(ext_tickers)))
    
    ext_dfs = {}
    if ext_tickers:
        raw_ext = yf.download(ext_tickers, start=START_DATE, end=END_DATE, progress=False)
        for t in ext_tickers:
            data = raw_ext[TYPE][t] if len(ext_tickers) > 1 else raw_ext[TYPE]
            if not data.dropna().empty: ext_dfs[t] = data.rename(t)

    def prepare_data(N):
        N = int(N)
        df_d = stock_data[[TYPE, "High", "Low", "Volume"]].copy()
        df_d['idx_p'] = df_d[TYPE] # Self-referential index
        for t, data in ext_dfs.items(): df_d = df_d.join(data, how='left')
        
        df_d = df_d.ffill().dropna()
        df_d["pct_r"] = df_d[TYPE].pct_change()
        ma20_p = df_d[TYPE].rolling(20).mean()
        ma20_v = df_d["Volume"].rolling(20).mean()
        tr = pd.concat([df_d["High"]-df_d["Low"], abs(df_d["High"]-df_d[TYPE].shift(1)), abs(df_d["Low"]-df_d[TYPE].shift(1))], axis=1).max(axis=1)
        df_d["vol_daily"] = df_d["pct_r"].rolling(60).std()
        
        df_d["r3"] = np.log(df_d[TYPE] / (df_d[TYPE].shift(3) + 1e-9))
        df_d["dt"] = (df_d[TYPE] - ma20_p) / (ma20_p + 1e-9)
        df_d["sigma_t"] = df_d["pct_r"].rolling(N).std()
        df_d["delta_sigma"] = df_d["sigma_t"] - df_d["sigma_t"].shift(5)
        df_d["natr"] = tr.rolling(N).mean() / (df_d[TYPE] + 1e-9)
        df_d["rv"] = df_d["Volume"] / (ma20_v + 1e-9)
        
        df_d = df_d.dropna()
        starts = np.arange(len(df_d) - N, -1, -N)[::-1]
        
        chunk_rows = []
        for i in range(1, len(starts)):
            block_prev = df_d.iloc[starts[i-1] : starts[i]]
            block_curr = df_d.iloc[starts[i] : starts[i] + N]
            c_ret = (block_prev[TYPE].iloc[-1] - block_prev[TYPE].iloc[0]) / (block_prev[TYPE].iloc[0] + 1e-9)
            
            row = {
                "Date": df_d.index[starts[i]], 
                "y": (block_curr[TYPE].iloc[-1] - block_curr[TYPE].iloc[0]) / (block_curr[TYPE].iloc[0] * block_curr["vol_daily"].iloc[0] * np.sqrt(N) + 1e-9),
                "f_r3": block_prev["r3"].iloc[-1], "f_dt": block_prev["dt"].iloc[-1], 
                "f_sigma": block_prev["sigma_t"].iloc[-1], "f_dsigma": block_prev["delta_sigma"].iloc[-1],
                "f_natr": block_prev["natr"].iloc[-1], "f_rv": block_prev["rv"].mean(), 
                "f_intensity": block_prev["rv"].mean() * abs(c_ret),
                "f_skew": np.nan_to_num(skew(block_prev["pct_r"]), nan=0.0), 
                "f_entropy": np.nan_to_num(-((block_prev["pct_r"]>0).mean()*np.log(np.clip((block_prev["pct_r"]>0).mean(),0.001,0.999)) + (1-(block_prev["pct_r"]>0).mean())*np.log(np.clip(1-(block_prev["pct_r"]>0).mean(),0.001,0.999))), nan=0.0)
            }
            for t in ext_tickers:
                if t in block_prev.columns:
                    row[f"ext_{t}"] = (block_prev[t].iloc[-1] - block_prev[t].iloc[0]) / (block_prev[t].iloc[0] + 1e-9)
            chunk_rows.append(row)
        
        df_chunks = pd.DataFrame(chunk_rows).replace([np.inf, -np.inf], np.nan).dropna()
        int_f = ["f_r3", "f_dt", "f_sigma", "f_dsigma", "f_natr", "f_rv", "f_intensity", "f_skew", "f_entropy"]
        ext_f = [f"ext_{t}" for t in ext_tickers if f"ext_{t}" in df_chunks.columns]
        return df_chunks, int_f, ext_f

    def evaluate_config(N, w):
        df, int_f, ext_f = prepare_data(N)
        if len(df) < HORIZON_BLOCKS + 5: return -1.0, -1.0, None
        
        horizon_indices = np.arange(len(df) - HORIZON_BLOCKS, len(df))
        mu_preds, actuals = [], []
        
        for idx in horizon_indices:
            train_df, test_df = df.iloc[:idx], df.iloc[[idx]]
            sCi, sCe = StandardScaler(), StandardScaler()
            Cti = sCi.fit_transform(train_df[int_f]) if int_f else np.empty((len(train_df), 0))
            Cte = sCe.fit_transform(train_df[ext_f]) if ext_f else np.empty((len(train_df), 0))
            Yt = train_df[["y"]].values.reshape(-1, 1)
            
            res = differential_evolution(neg_log_likelihood_dual, bounds=[(np.log(0.1), np.log(10)), (np.log(0.1), np.log(10)), (np.log(0.01), np.log(10)), (np.log(0.001), np.log(5))], args=(Cti, Cte, w, Yt), popsize=POPSIZE, tol=0.05, seed=RANDOM_SEED)
            li, le, sigma, sn = np.exp(res.x)
            
            Csi = sCi.transform(test_df[int_f]) if int_f else np.empty((1, 0))
            Cse = sCe.transform(test_df[ext_f]) if ext_f else np.empty((1, 0))
            mu_s, std_s = gaussian_process_dual(Cti, Csi, Cte, Cse, li, le, sigma, sn, w, Yt)
            mu_preds.append(mu_s[0]); actuals.append(test_df["y"].values[0])
            
        hr = np.mean(np.sign(mu_preds) == np.sign(actuals)) * 100
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(mu_preds))**2))
        return hr, rmse, {"mu": mu_preds, "actual": actuals, "std": std_s[0]}

    # Optimization Logic
    N_range = [5, 10, 20, 30, 40, 60]
    W_range = [0.0, 0.5, 1.0] if ext_tickers else [1.0]
    grid = [(n, w) for n in N_range for w in W_range]
    
    best_score, best_params, best_stats = -1.0, None, None
    for n, w in grid:
        hr, rmse, report = evaluate_config(n, w)
        if hr == -1.0: continue
        score = hr / (rmse + 1e-9) if hr >= 70 else -1.0
        if score > best_score:
            best_score, best_params, best_stats = score, (n, w), (hr, rmse, report)
    
    # Fallback to best Hit Rate if threshold not met
    if best_params is None:
        best_hr = -1.0
        for n, w in grid:
            hr, rmse, report = evaluate_config(n, w)
            if hr > best_hr:
                best_hr, best_params, best_stats = hr, (n, w), (hr, rmse, report)
        best_score = best_hr / (best_stats[1] + 1e-9)

    # Future Forecast with Champions
    best_N, best_W = best_params
    final_df, int_f, ext_f = prepare_data(best_N)
    
    # 100% Accurate Indicators for the upcoming chunk (Today forward)
    # Context comes from the block that just finished (Today-N to Today)
    df_d = stock_data[[TYPE, "High", "Low", "Volume"]].copy()
    df_d["pct_r"] = df_d[TYPE].pct_change()
    latest_block = df_d.tail(int(best_N))
    ma20_p, ma20_v = df_d[TYPE].rolling(20).mean().iloc[-1], df_d["Volume"].rolling(20).mean().iloc[-1]
    tr = pd.concat([latest_block["High"]-latest_block["Low"], abs(latest_block["High"]-latest_block[TYPE].shift(1))], axis=1).max(axis=1)
    
    # Internal context
    c_ret = (latest_block[TYPE].iloc[-1] - latest_block[TYPE].iloc[0]) / (latest_block[TYPE].iloc[0] + 1e-9)
    f_r3 = np.log(latest_block[TYPE].iloc[-1] / (latest_block[TYPE].iloc[-4] + 1e-9))
    f_dt = (latest_block[TYPE].iloc[-1] - ma20_p) / (ma20_p + 1e-9)
    f_sigma = latest_block["pct_r"].std()
    f_dsigma = f_sigma - df_d["pct_r"].shift(5).tail(int(best_N)).std()
    f_natr = tr.mean() / (latest_block[TYPE].iloc[-1] + 1e-9)
    f_rv = latest_block["Volume"].mean() / (df_d["Volume"].mean() + 1e-9)
    f_intensity = f_rv * abs(c_ret)
    f_skew = np.nan_to_num(skew(latest_block["pct_r"].dropna()), nan=0.0)
    p_up = np.clip((latest_block["pct_r"] > 0).mean(), 0.001, 0.999)
    f_entropy = -p_up * np.log(p_up) - (1-p_up) * np.log(1-p_up)
    
    int_row = [f_r3, f_dt, f_sigma, f_dsigma, f_natr, f_rv, f_intensity, f_skew, f_entropy]
    
    # External context
    ext_row = []
    if ext_tickers:
        for t in ext_tickers:
            if t in df_d.columns:
                ext_row.append((df_d[t].iloc[-1] - df_d[t].iloc[-int(best_N)]) / (df_d[t].iloc[-int(best_N)] + 1e-9))
            else:
                ext_row.append(0.0)

    sCi, sCe = StandardScaler(), StandardScaler()
    Cti = sCi.fit_transform(final_df[int_f]) if int_f else np.empty((len(final_df), 0))
    Cte = sCe.fit_transform(final_df[ext_f]) if ext_f else np.empty((len(final_df), 0))
    Yt = final_df[["y"]].values.reshape(-1, 1)
    
    res = differential_evolution(neg_log_likelihood_dual, bounds=[(np.log(0.1), np.log(10)), (np.log(0.1), np.log(10)), (np.log(0.01), np.log(10)), (np.log(0.001), np.log(5))], args=(Cti, Cte, best_W, Yt), popsize=POPSIZE, tol=0.05, seed=RANDOM_SEED)
    li, le, sigma, sn = np.exp(res.x)
    
    mu_f, std_f = gaussian_process_dual(Cti, sCi.transform(np.array([int_row])), Cte, sCe.transform(np.array([ext_row])) if ext_f else np.empty((1,0)), li, le, sigma, sn, best_W, Yt)

    return {
        'Stock': ticker,
        'Sectors': sector_codes,
        'Opt. N': int(best_N),
        'Eff. Index': f"{best_score:.2f}",
        'Hit Rate': f"{best_stats[0]:.1f}%",
        'Pred Mean (Z)': f"{mu_f[0]:+.2f}",
        '95% CI (±Z)': f"{1.96 * std_f[0]:.2f}",
        'Signal': 'UP' if mu_f[0] > 0 else 'DOWN'
    }

# ==========================================
# 3. Main Scanner Execution
# ==========================================
if __name__ == "__main__":
    print(f"==========================================")
    print(f"PORTFOLIO GP SCANNER: {len(PORTFOLIO_INPUT)} Stocks")
    print(f"==========================================")
    
    # Process portfolio in parallel (Stock level)
    results = Parallel(n_jobs=-1, backend="threading")(delayed(process_single_stock)(t, s) for t, s in PORTFOLIO_INPUT)
    
    # Compile Table
    clean_results = [r for r in results if r is not None]
    df_portfolio = pd.DataFrame(clean_results)
    
    print("\n\nFINAL PORTFOLIO INTELLIGENCE TABLE:")
    print(df_portfolio.to_string(index=False))
    
    # Save results
    df_portfolio.to_csv('portfolio_scan_results.csv', index=False)
    print(f"\nResults saved to portfolio_scan_results.csv")
