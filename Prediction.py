"""Degradation Prediction and Analysis Tool.

Fits empirical and kinetic degradation models to every batch of every Excel dataset and reports:
  * goodness of fit (R2, adjusted R2, RMSE), information criteria (AICc, BIC) and leave-one-out
    cross validation error,
  * Akaike weights and a model averaged time to the degradation threshold,
  * 95% confidence intervals for parameters (from the covariance matrix) and for the time to the
    threshold plus 95% prediction bands (residual bootstrap),
  * an extra sum of squares F-test of each batch against the control batch, with Holm correction,
  * residual diagnostics and publication ready figures (PNG and PDF) and tables (CSV and LaTeX).

Modelling assumptions: degradation is cumulative, so every model is constrained to be non-decreasing,
and the kinetic models are bounded by a plateau of at most 100% (or the largest observation).
Bootstrap and F-test results are approximate when only a few points per batch are available.

Install: pip install -r requirements.txt
"""

import os
import traceback
import warnings

import matplotlib
matplotlib.use('Agg')  # figures are only written to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

# Plot text: Times New Roman, black, sizes 16 (titles), 14 (axis labels), 12 (everything else)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14, 'axes.labelweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'pdf.fonttype': 42, 'savefig.dpi': 300,
})

SEED = 42          # fixed seed so bootstrap results are reproducible
EPS = 1e-9         # floor that keeps log, power and rate parameters positive
MAX_EXP = 50.0     # exponent clip so exp() cannot overflow
RULE = '=' * 60
INF = np.inf


# ==============================================================================================
# Model functions. The same function is used to fit, to predict and to plot.
# ==============================================================================================
def _exp(z): return np.exp(np.clip(z, -MAX_EXP, MAX_EXP))
def _linear(x, m, c): return m * x + c
def _logarithmic(x, a, b, c): return a * np.log(x + b) + c
def _power_law(x, a, b, c): return a * np.power(np.maximum(x, 0.0), b) + c
def _exponential(x, a, b, c): return a * _exp(b * x) + c
def _first_order(x, A, k): return A * (1.0 - _exp(-k * x))
def _weibull(x, A, L, b): return A * (1.0 - _exp(-np.power(np.maximum(x, 0.0) / L, b)))
def _gompertz(x, A, b, c): return A * _exp(-b * _exp(-c * x))
def _logistic(x, A, k, t0): return A / (1.0 + _exp(-k * (x - t0)))


# name: (function, parameter names, equation for the plot, line colour)
MODELS = {
    'Linear': (_linear, ('m', 'c'), 'y = m·x + c', 'blue'),
    'Logarithmic': (_logarithmic, ('a', 'b', 'c'), 'y = a·ln(x + b) + c', 'green'),
    'Power Law': (_power_law, ('a', 'b', 'c'), 'y = a·x^b + c', 'purple'),
    'Exponential': (_exponential, ('a', 'b', 'c'), 'y = a·exp(b·x) + c', 'brown'),
    'First Order': (_first_order, ('A', 'k'), 'y = A·(1 − exp(−k·x))', 'darkorange'),
    'Weibull': (_weibull, ('A', 'L', 'b'), 'y = A·(1 − exp(−(x/L)^b))', 'teal'),
    'Gompertz': (_gompertz, ('A', 'b', 'c'), 'y = A·exp(−b·exp(−c·x))', 'magenta'),
    'Logistic': (_logistic, ('A', 'k', 't0'), 'y = A / (1 + exp(−k·(x − t0)))', 'olive'),
}


def _setup(name, x, y):
    """Starting points and bounds for one model. x is sorted ascending."""
    tx, ymax, y0, ymin = max(x.max(), EPS), y.max(), y[0], y.min()
    rng, cap = max(np.ptp(y), EPS), max(100.0, 1.05 * y.max())
    slope = max((y[-1] - y[0]) / max(x[-1] - x[0], EPS), 0.0)
    a0, a1 = rng / np.expm1(2.0), rng / np.expm1(1.0)
    top = min(cap, 1.3 * ymax)
    if name == 'Linear':
        starts, lo, hi = [[slope, y0], [0.1, 0.0]], [0, -INF], [INF, INF]
    elif name == 'Logarithmic':
        starts, lo, hi = [[rng, 1.0, ymin], [10.0, 1.0, 0.0], [rng, tx, 0.0]], [0, EPS, -INF], [INF, 1e4, INF]
    elif name == 'Power Law':
        starts = [[slope, 1.0, y0], [1.0, 0.5, 0.0], [rng / np.sqrt(tx), 0.5, y0]]
        lo, hi = [0, 0.05, -INF], [INF, 5.0, INF]
    elif name == 'Exponential':
        starts = [[a0, 2.0 / tx, y0 - a0], [1.0, 0.01, y0], [a1, 1.0 / tx, y0 - a1]]
        lo, hi = [0, 0, -INF], [INF, 1.0, INF]
    elif name == 'First Order':
        starts, lo, hi = [[top, 2.0 / tx], [cap, 0.5 / tx], [ymax, 5.0 / tx]], [EPS, EPS], [cap, 10.0]
    elif name == 'Weibull':
        starts = [[top, tx, 1.5], [cap, 2.0 * tx, 1.0], [ymax, 0.5 * tx, 3.0]]
        lo, hi = [EPS, EPS, 0.1], [cap, 1e5, 6.0]
    elif name == 'Gompertz':
        starts = [[top, 3.0, 3.0 / tx], [cap, 5.0, 1.0 / tx], [ymax, 1.0, 6.0 / tx]]
        lo, hi = [EPS, 1e-3, EPS], [cap, 50.0, 10.0]
    else:  # Logistic
        starts = [[top, 6.0 / tx, tx / 2], [cap, 3.0 / tx, tx], [ymax, 12.0 / tx, tx / 3]]
        lo, hi = [EPS, EPS, 0.0], [cap, 10.0, 1e5]
    lo, hi = np.array(lo, float), np.array(hi, float)
    return [np.clip(p, lo + 1e-12, hi - 1e-12) for p in starts], lo, hi


def _curve_fit(func, x, y, starts, lo, hi, nfev=2000):
    """Best bounded least squares fit over the starting points: (sse, popt, pcov) or None."""
    best = None
    for p0 in starts:
        try:
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore', OptimizeWarning)
                popt, pcov = curve_fit(func, x, y, p0=p0, bounds=(lo, hi), max_nfev=nfev)
                sse = float(np.sum((y - func(x, *popt)) ** 2))
        except (RuntimeError, ValueError):  # no convergence or non-finite residuals
            continue
        if np.isfinite(sse) and (best is None or sse < best[0]):
            best = (sse, popt, pcov)
    return best


def _metrics(y, sse, n, p):
    k = p + 1  # fitted parameters plus the error variance
    r2 = 1.0 - sse / float(np.sum((y - y.mean()) ** 2))
    ll = n * np.log(max(sse, 1e-12) / n)
    return {
        'r2': r2, 'adj_r2': 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan,
        'rmse': float(np.sqrt(sse / n)), 'bic': ll + k * np.log(n), 'n': n, 'k': k,
        'aicc': ll + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan,
    }


def _fit_model(name, x, y):
    """Fit one model. Returns a dict with parameters, intervals and metrics, or None."""
    func, pnames = MODELS[name][0], MODELS[name][1]
    starts, lo, hi = _setup(name, x, y)
    best = _curve_fit(func, x, y, starts, lo, hi)
    if best is None:
        return None
    sse, popt, pcov = best
    n, p = len(x), len(popt)
    m = _metrics(y, sse, n, p)
    if m['r2'] < 0:
        return None
    se = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(p, np.nan)
    tval = stats.t.ppf(0.975, n - p) if n > p else np.nan
    return {'popt': popt, 'ci_lo': popt - tval * se, 'ci_hi': popt + tval * se, 'pnames': pnames,
            'sse': sse, 'metrics': m, 'lo': lo, 'hi': hi}


def _loocv_rmse(name, x, y, popt):
    """Leave one out cross validation RMSE (NaN when too few points remain)."""
    func, p, errs = MODELS[name][0], len(popt), []
    if len(x) < p + 2:
        return np.nan
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        starts, lo, hi = _setup(name, x[keep], y[keep])
        p0 = np.clip(popt, lo + 1e-12, hi - 1e-12)  # warm start from the full fit, then the usual starts if it fails
        best = (_curve_fit(func, x[keep], y[keep], [p0], lo, hi, 300)
                or _curve_fit(func, x[keep], y[keep], starts[:1], lo, hi, 300))
        if best is None:
            return np.nan
        with np.errstate(all='ignore'):
            errs.append(float(func(x[i:i + 1], *best[1])[0] - y[i]))
    return float(np.sqrt(np.mean(np.square(errs))))


def _days_to_threshold(days, pred, threshold):
    """First time the curve reaches the threshold (linear interpolation), or NaN if it never does."""
    hit = pred >= threshold
    if not hit.any():
        return np.nan
    i = int(hit.argmax())
    if i == 0:
        return float(days[0])
    return float(np.interp(threshold, pred[i - 1:i + 1], days[i - 1:i + 1]))


def _bootstrap(name, x, y, fit, grid, threshold, n_boot, rng):
    """Residual bootstrap. Returns (95% band on the grid or None, time to threshold draws)."""
    func, popt = MODELS[name][0], fit['popt']
    fitted = func(x, *popt)
    resid = (y - fitted) * np.sqrt(len(y) / max(len(y) - len(popt), 1))
    curves, days = [], []
    for _ in range(n_boot):
        yb = fitted + rng.choice(resid, size=len(y))
        best = _curve_fit(func, x, yb, [popt], fit['lo'], fit['hi'], 300)
        if best is None:
            continue
        with np.errstate(all='ignore'):
            curve = func(grid, *best[1])
        if not np.isfinite(curve).all():
            continue
        curves.append(curve)
        d = _days_to_threshold(grid, curve, threshold)
        days.append(INF if np.isnan(d) else d)
    if len(curves) < 20:
        return None, np.array([])
    return np.percentile(curves, [2.5, 97.5], axis=0), np.array(days)


def _interval(draws):
    return tuple(np.quantile(draws, [0.025, 0.975], method='nearest')) if len(draws) else (np.nan, np.nan)


def _f_test(name, ctrl, other):
    """Extra sum of squares F-test: one shared curve versus separate curves. Returns (F, df1, df2, p)."""
    f1, f2 = ctrl['models'].get(name), other['models'].get(name)
    if f1 is None or f2 is None:
        return None
    x, y = np.r_[ctrl['x'], other['x']], np.r_[ctrl['y'], other['y']]
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    starts, lo, hi = _setup(name, x, y)
    pooled = _curve_fit(MODELS[name][0], x, y, starts, lo, hi)
    p, sse_sep = len(f1['popt']), f1['sse'] + f2['sse']
    dof2 = len(x) - 2 * p
    if pooled is None or dof2 <= 0 or sse_sep < 1e-12:
        return None
    F = max((pooled[0] - sse_sep) / p / (sse_sep / dof2), 0.0)
    return F, p, dof2, float(stats.f.sf(F, p, dof2))


def _holm(pvals):
    order = np.argsort(pvals)
    adj, run = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def _fmt_days(v, nd=2):
    return 'Not Reached' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def _fmt_ci(lo, hi, nd=2):
    return 'N/A' if lo is None or np.isnan(lo) else f'[{_fmt_days(lo, nd)}, {_fmt_days(hi, nd)}]'


def _fmt(v, nd=4):
    return 'N/A' if v is None or not np.isfinite(v) else f'{v:.{nd}f}'


def _safe(text, extra=''):
    return ''.join(c for c in text if c.isalnum() or c in extra)


def _tex(text):
    for a, b in (('\\', ''), ('%', r'\%'), ('&', r'\&'), ('_', r'\_'), ('#', r'\#')):
        text = text.replace(a, b)
    return text


def _param_text(fit):
    return '; '.join(f'{n}={v:.4g} [{_fmt(lo, 4)}, {_fmt(hi, 4)}]'
                     for n, v, lo, hi in zip(fit['pnames'], fit['popt'], fit['ci_lo'], fit['ci_hi']))


class DegradationAnalysis:
    """Runs the full workflow for every dataset and writes figures, tables and a report."""

    def __init__(self, datasets_config, selected_models, threshold=50.0, n_boot=300):
        if not datasets_config: raise ValueError("Dataset configuration cannot be empty.")
        if not selected_models: raise ValueError("A list of models to run must be provided.")
        self.datasets_config, self.selected_models = datasets_config, list(selected_models)
        self.threshold, self.n_boot = float(threshold), int(n_boot)
        self.rng = np.random.default_rng(SEED)
        self.dataframes, self.results, self.batches, self.comparisons = {}, {}, {}, {}

        self.output_dir = os.path.dirname(os.path.abspath(datasets_config[0]['path']))
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n✅ All outputs will be saved to '{self.graphs_dir}' and '{self.results_dir}'")

    # ==========================================================================================
    # Data
    # ==========================================================================================
    def _preprocess_data(self, file_path):
        base = os.path.basename(file_path)
        print(f"\nPreprocessing data from '{base}'...")
        try:
            with pd.ExcelFile(file_path) as xl:
                df = xl.parse('t-DR' if 't-DR' in xl.sheet_names else xl.sheet_names[0])
        except Exception as e:
            raise OSError(f"Could not read the Excel file '{file_path}'. Error: {e}") from e

        df.columns = df.columns.astype(str).str.strip()
        lower = {c.lower(): c for c in df.columns}
        tcol = next((lower[k] for k in ('day', 'days', 'time', 'hours') if k in lower), None)
        if tcol is None:
            raise ValueError(f"Could not find a time column (Day, Days, Time or Hours) in {base}.")

        raw = {'Day': pd.to_numeric(df[tcol], errors='coerce') / (24 if 'hour' in tcol.lower() else 1)}
        for col in df.columns:  # every other column with at least 3 numeric values is a batch
            if col != tcol:
                s = pd.to_numeric(df[col], errors='coerce')
                if s.notna().sum() >= 3:
                    raw[f'{col}_degradation'] = s.clip(lower=0)
        if len(raw) == 1:
            raise ValueError(f"No numeric batch columns were found in {base}.")

        # Sorted by time, every reading kept. Gaps inside a series are filled by linear interpolation
        # on the Day values; leading and trailing gaps stay empty so no data is invented at the edges.
        df = pd.DataFrame(raw).dropna(subset=['Day']).sort_values('Day', kind='stable').reset_index(drop=True)
        for col in df.columns[1:]:
            ok = df[col].notna()
            if ok.any():
                gap = df[col].isna() & df['Day'].between(df.loc[ok, 'Day'].iloc[0], df.loc[ok, 'Day'].iloc[-1])
                df.loc[gap, col] = np.interp(df.loc[gap, 'Day'], df.loc[ok, 'Day'], df.loc[ok, col])
        if len(df) < 3:
            raise ValueError(f"Insufficient data points (<3) in {base}.")

        out = os.path.join(self.results_dir, f'{os.path.splitext(base)[0].lower()}_processed.csv')
        df.to_csv(out, index=False)
        print(f"Successfully processed {len(raw) - 1} batches. Saved to {out}")
        return df

    # ==========================================================================================
    # Modelling
    # ==========================================================================================
    def _analyse_dataset(self, info):
        name = info['name']
        df = self._preprocess_data(info['path'])
        self.dataframes[name] = df
        self.batches[name] = [c[:-len('_degradation')] for c in df.columns[1:]]
        results = self.results.setdefault(name, {})
        grid = np.linspace(0, max(2000.0, 1.25 * df['Day'].max() + 5), 2000)

        print(f"\n{RULE}\nCreating models for '{name}'\n{RULE}")
        for batch in self.batches[name]:
            sub = df[['Day', f'{batch}_degradation']].dropna()
            x, y = sub['Day'].to_numpy(), sub[f'{batch}_degradation'].to_numpy()
            print(f"\nProcessing {batch}")
            if len(x) < 3 or np.ptp(y) == 0:
                print("    Skipped: fewer than 3 points or no variation."); continue
            rec = results[batch] = self._fit_batch(x, y, grid)
            for model in rec['models']:
                self._plot_fit(rec, grid, batch, name, model)
            if rec['models']:
                self._plot_overlay(rec, grid, batch, name)
                self._plot_residuals(rec, batch, name)
                best = rec['best']
                print(f"  Best model for {batch}: {best} ({rec['criterion']} weight {rec['models'][best]['weight']:.3f}, "
                      f"R² = {rec['models'][best]['metrics']['r2']:.4f})")
        self._compare_batches(name)

    def _fit_batch(self, x, y, grid):
        rec = {'x': x, 'y': y, 'models': {}}
        for model in self.selected_models:
            fit = _fit_model(model, x, y)
            if fit is None:
                print(f"    Failed to fit {model} model."); continue
            with np.errstate(all='ignore'):
                pred = MODELS[model][0](grid, *fit['popt'])
            if not np.isfinite(pred).all():
                print(f"    {model}: extrapolation is not finite, skipped."); continue
            fit.update(pred=pred, d50=_days_to_threshold(grid, pred, self.threshold),
                       loocv=_loocv_rmse(model, x, y, fit['popt']), band=None, draws=np.array([]))
            rec['models'][model] = fit
        if not rec['models']:
            rec.update(best=None, criterion='AICc', avg=None)
            return rec

        # ranking: AICc when it exists for every model, otherwise BIC, so the criterion is consistent
        crit = 'aicc' if all(np.isfinite(f['metrics']['aicc']) for f in rec['models'].values()) else 'bic'
        vals = np.array([f['metrics'][crit] for f in rec['models'].values()])
        delta = vals - vals.min()
        weights = np.exp(-delta / 2.0) / np.exp(-delta / 2.0).sum()
        for (model, fit), d, w in zip(rec['models'].items(), delta, weights):
            fit.update(delta=float(d), weight=float(w))
            m = fit['metrics']
            print(f"    {model}: R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}, {crit.upper()}={m[crit]:.2f}, "
                  f"weight={w:.3f}, LOOCV RMSE={_fmt(fit['loocv'])}")
        rec['criterion'] = 'BIC' if crit == 'bic' else 'AICc'
        rec['best'] = min(rec['models'], key=lambda m: rec['models'][m]['delta'])

        if self.n_boot > 0:
            for model, fit in rec['models'].items():
                fit['band'], fit['draws'] = _bootstrap(model, x, y, fit, grid, self.threshold, self.n_boot, self.rng)
        rec['avg'] = self._average(rec)
        return rec

    def _average(self, rec):
        """Model averaged time to the threshold: Akaike weighted pool of the bootstrap draws."""
        models = rec['models']
        pooled = [self.rng.choice(f['draws'], size=int(round(f['weight'] * self.n_boot)))
                  for f in models.values() if len(f['draws'])]
        pooled = np.concatenate(pooled) if pooled else np.array([])
        if len(pooled):
            lo, hi = _interval(pooled)
            return {'d50': float(np.median(pooled)), 'lo': float(lo), 'hi': float(hi)}
        d = np.array([f['d50'] for f in models.values()])
        w = np.array([f['weight'] for f in models.values()])
        return {'d50': float(np.sum(w * d)) if np.isfinite(d).all() else np.nan, 'lo': np.nan, 'hi': np.nan}

    def _compare_batches(self, dataset):
        batches = [b for b in self.batches[dataset] if b in self.results[dataset]]
        if len(batches) < 2:
            return
        ctrl, rows = batches[0], []
        for model in self.selected_models:
            part = []
            for b in batches[1:]:
                t = _f_test(model, self.results[dataset][ctrl], self.results[dataset][b])
                if t:
                    part.append((b, *t))
            adj = _holm([t[4] for t in part]) if part else []
            for (b, F, d1, d2, p), pa in zip(part, adj):
                rows.append({'Model': model, 'Control': ctrl, 'Batch': b, 'F': round(F, 4), 'df1': d1, 'df2': d2,
                             'p': round(p, 6), 'p (Holm)': round(float(pa), 6), 'Different from control': bool(pa < 0.05)})
        self.comparisons[dataset] = pd.DataFrame(rows)
        if rows:
            self.comparisons[dataset].to_csv(
                os.path.join(self.results_dir, f'batch_comparison_{_safe(dataset)}.csv'), index=False)

    # ==========================================================================================
    # Figures
    # ==========================================================================================
    def _save(self, fig, stem, pdf=False):
        fig.tight_layout()
        fig.savefig(os.path.join(self.graphs_dir, f'{stem}.png'), bbox_inches='tight')
        if pdf:
            fig.savefig(os.path.join(self.graphs_dir, f'{stem}.pdf'), bbox_inches='tight')

    def _limit(self, rec, grid, d50):
        limit = np.ceil(1.25 * rec['x'].max() / 5) * 5
        if np.isfinite(d50): limit = max(limit, np.ceil(1.05 * d50 / 5) * 5)
        return min(limit, grid[-1])

    def _axes_style(self, ax, ylabel='Degradation (%)'):
        ax.set(xlabel='Time (Days)', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    def _plot_fit(self, rec, grid, batch, dataset, model):
        fit, x, y = rec['models'][model], rec['x'], rec['y']
        eq, color = MODELS[model][2:]
        d50, last = fit['d50'], x.max()
        limit = self._limit(rec, grid, d50)
        keep = grid <= limit
        g, p = grid[keep], fit['pred'][keep]
        k = np.searchsorted(g, last, side='right')
        m = fit['metrics']

        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            ax.scatter(x, y, color='red', s=50, zorder=5, label='Actual Data')
            if fit['band'] is not None:
                ax.fill_between(g, fit['band'][0][keep], fit['band'][1][keep], color=color, alpha=0.15,
                                label='95% bootstrap band')
            ax.plot(g[:k], p[:k], color=color, linewidth=2.5, label=f"{model} Model (R² = {m['r2']:.4f})")
            ax.plot(g[max(k - 1, 0):], p[max(k - 1, 0):], color=color, linewidth=2.5, linestyle='dashed')
            ax.axvline(last, color='gray', linestyle='dotted', linewidth=2, label=f'End of Actual Data ({last:.0f} days)')
            ax.axhline(self.threshold, color='black', linestyle='dashed', alpha=0.7,
                       label=f'{self.threshold:g}% Degradation Threshold')
            if np.isfinite(d50):
                lo, hi = _interval(fit['draws'])
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.axvspan(lo, min(hi, limit), color='orange', alpha=0.15, label='95% interval of time to threshold')
                ax.axvline(d50, color='orange', linestyle='dotted', alpha=0.9)
                # degradation rises with time, so the area below and right of the crossing (or above and left
                # of it, when the line is in the right half) is free of the curve and the data
                if d50 > limit / 2:
                    ax.text(d50 - 0.01 * limit, 100, f'{d50:.1f} days', fontweight='bold', rotation=90, ha='right', va='top')
                else:
                    ax.text(d50 + 0.01 * limit, 5, f'{d50:.1f} days', fontweight='bold', rotation=90, ha='left', va='bottom')
            self._axes_style(ax)
            ax.set(title=f"{batch} | {dataset}\n{model} Model Extrapolation", ylim=(0, 105), xlim=(0, limit))
            ax.legend(loc='upper left')
            info = (f"Model: {eq}\nR² = {m['r2']:.4f}   RMSE = {m['rmse']:.4f}\n"
                    f"{rec['criterion']} = {m['bic' if rec['criterion'] == 'BIC' else 'aicc']:.2f}   "
                    f"weight = {fit['weight']:.3f}")
            ax.text(0.98, 0.02, info, transform=ax.transAxes, va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            self._save(fig, f"{_safe(batch, ' _')}_{_safe(dataset)}_{model}_model")
        finally:
            plt.close(fig)

    def _plot_overlay(self, rec, grid, batch, dataset):
        x, y, models = rec['x'], rec['y'], rec['models']
        limit = self._limit(rec, grid, models[rec['best']]['d50'])
        keep = grid <= limit
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            ax.scatter(x, y, color='red', s=50, zorder=5, label='Actual Data')
            for model, fit in models.items():
                ax.plot(grid[keep], fit['pred'][keep], color=MODELS[model][3],
                        linewidth=2.5 if model == rec['best'] else 1.5, label=f"{model} (w = {fit['weight']:.2f})")
            ax.axvline(x.max(), color='gray', linestyle='dotted', linewidth=2)
            ax.axhline(self.threshold, color='black', linestyle='dashed', alpha=0.7)
            self._axes_style(ax)
            ax.set(title=f"{batch} | {dataset}\nAll Models Compared by {rec['criterion']} Weight",
                   ylim=(0, 105), xlim=(0, limit))
            ax.legend(loc='upper left')
            self._save(fig, f"{_safe(batch, ' _')}_{_safe(dataset)}_all_models", pdf=True)
        finally:
            plt.close(fig)

    def _plot_residuals(self, rec, batch, dataset):
        best = rec['best']
        resid = rec['y'] - MODELS[best][0](rec['x'], *rec['models'][best]['popt'])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        try:
            a1.scatter(MODELS[best][0](rec['x'], *rec['models'][best]['popt']), resid, color='red', s=40)
            a1.axhline(0, color='black', linestyle='dashed')
            a1.set(title='Residuals versus Fitted', xlabel='Fitted (%)', ylabel='Residual (%)')
            (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist='norm')
            a2.scatter(osm, osr, color='red', s=40)
            a2.plot(osm, slope * osm + icpt, color='black')
            a2.set(title='Normal Q-Q Plot', xlabel='Theoretical Quantiles', ylabel='Ordered Residuals')
            if len(resid) >= 3 and np.ptp(resid) > 0:
                a2.text(0.03, 0.97, f'Shapiro-Wilk p = {stats.shapiro(resid)[1]:.3f}', transform=a2.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            fig.suptitle(f'{batch} | {dataset} | {best}', fontsize=16)
            self._save(fig, f"{_safe(batch, ' _')}_{_safe(dataset)}_residuals", pdf=True)
        finally:
            plt.close(fig)

    def create_performance_heatmaps(self):
        print(f"\n{RULE}\nCREATING PERFORMANCE HEATMAPS\n{RULE}")
        for dataset, results in self.results.items():
            batches, shape = self.batches[dataset], (len(self.selected_models), len(self.batches[dataset]))
            grids = {k: np.full(shape, np.nan) for k in ('r2', 'rmse', 'weight')}
            for i, model in enumerate(self.selected_models):
                for j, batch in enumerate(batches):
                    fit = results.get(batch, {}).get('models', {}).get(model)
                    if fit:
                        grids['r2'][i, j], grids['rmse'][i, j] = fit['metrics']['r2'], fit['metrics']['rmse']
                        grids['weight'][i, j] = fit['weight']
            safe = _safe(dataset)
            self._plot_heatmap(grids['r2'], batches, f'{dataset}\nR² Performance', f'heatmap_{safe}_R2')
            self._plot_heatmap(grids['rmse'], batches, f'{dataset}\nRMSE Performance', f'heatmap_{safe}_RMSE', 'RdYlGn_r')
            self._plot_heatmap(grids['weight'], batches, f'{dataset}\nModel Weight', f'heatmap_{safe}_Weight')

    def _plot_heatmap(self, data, batches, title, stem, cmap='RdYlGn'):
        fig, ax = plt.subplots(figsize=(10, 5 + 0.3 * len(self.selected_models)))
        try:
            if np.isnan(data).all():
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set(xticks=[], yticks=[], title=title)
            else:
                ax.imshow(data, cmap=cmap, aspect='auto')
                ax.set(xticks=np.arange(len(batches)), yticks=np.arange(len(self.selected_models)),
                       xticklabels=batches, yticklabels=self.selected_models, title=title,
                       xlabel='Batches', ylabel='Models')
                ax.xaxis.labelpad = ax.yaxis.labelpad = 15
                plt.setp(ax.get_xticklabels(), rotation=15, ha='right', rotation_mode='anchor')
                for (i, j), v in np.ndenumerate(data):
                    if not np.isnan(v): ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=12)
                ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)  # cell borders
                ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
                ax.grid(which='minor', color='black', linestyle='solid', linewidth=2)
                ax.tick_params(which='minor', bottom=False, left=False)
            self._save(fig, stem, pdf=True)
        finally:
            plt.close(fig)
        print(f"Heatmap saved: {stem}.png")

    # ==========================================================================================
    # Tables and report
    # ==========================================================================================
    def create_summary_tables(self):
        print(f"\n{RULE}\nDEGRADATION ANALYSIS SUMMARY\n{RULE}")
        t = f'{self.threshold:g}%'
        for dataset, results in self.results.items():
            print(f"\nSUMMARY FOR: '{dataset}'")
            allrows = []
            for model in self.selected_models:
                print(f"\n{model.upper()} MODEL RESULTS")
                rows = []
                for batch in self.batches[dataset]:
                    fit = results.get(batch, {}).get('models', {}).get(model)
                    m = fit['metrics'] if fit else {}
                    lo, hi = _interval(fit['draws']) if fit else (np.nan, np.nan)
                    days = 'N/A' if not fit else (_fmt(fit['d50'], 4) if np.isfinite(fit['d50']) else 'Not Reached')
                    rows.append({'Batch': batch, f'Days to {t}': days, 'CI low': _fmt(lo, 2), 'CI high': _fmt(hi, 2),
                                 'R²': _fmt(m.get('r2')), 'Adjusted R²': _fmt(m.get('adj_r2')), 'RMSE': _fmt(m.get('rmse')),
                                 'AICc': _fmt(m.get('aicc'), 2), 'BIC': _fmt(m.get('bic'), 2),
                                 'LOOCV RMSE': _fmt(fit['loocv']) if fit else 'N/A',
                                 'Weight': _fmt(fit['weight'], 3) if fit else 'N/A'})
                    if fit:
                        allrows.append({'Dataset': dataset, 'Model': model, 'Points': m['n'],
                                        'Parameters (95% CI)': _param_text(fit), 'Delta': _fmt(fit['delta'], 2), **rows[-1]})
                table = pd.DataFrame(rows)
                print(table[['Batch', f'Days to {t}', 'R²', 'RMSE', 'AICc', 'Weight']].to_string(index=False))
                table.to_csv(os.path.join(self.results_dir, f'summary_{_safe(dataset)}_{model}.csv'), index=False)
            pd.DataFrame(allrows).to_csv(os.path.join(self.results_dir, f'all_results_{_safe(dataset)}.csv'), index=False)
            self._best_table(dataset)
            print(f"Results for '{dataset}' saved to CSV and LaTeX files.")

    def _best_table(self, dataset):
        t, rows = f'{self.threshold:g}%', []
        for batch in self.batches[dataset]:
            rec = self.results[dataset].get(batch)
            if not rec or not rec['models']:
                continue
            fit, avg, last = rec['models'][rec['best']], rec['avg'], rec['x'].max()
            d = avg['d50'] if np.isfinite(avg['d50']) else fit['d50']
            extent = 'Not Reached' if not np.isfinite(d) else (
                'Within data' if d <= last else f'Extrapolated {d / last:.1f}x the last day')
            rows.append({'Batch': batch, 'Points': len(rec['x']), 'Best model': rec['best'],
                         'R²': _fmt(fit['metrics']['r2']), 'RMSE': _fmt(fit['metrics']['rmse']),
                         'Weight': _fmt(fit['weight'], 3), f'Days to {t} (best model)': _fmt_days(fit['d50']),
                         'CI (best model)': _fmt_ci(*_interval(fit['draws'])),
                         f'Days to {t} (model averaged)': _fmt_days(avg['d50']),
                         'CI (model averaged)': _fmt_ci(avg['lo'], avg['hi']), 'Extent': extent})
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(self.results_dir, f'best_models_{_safe(dataset)}.csv'), index=False)
        if table.empty:
            return
        cols = ['Batch', 'Best model', 'R²', 'RMSE', 'Weight', f'Days to {t} (model averaged)', 'CI (model averaged)']
        tex = lambda s: _tex(str(s)).replace('R²', 'R$^2$')
        with open(os.path.join(self.results_dir, f'best_models_{_safe(dataset)}.tex'), 'w', encoding='utf-8') as f:
            f.write('\\begin{tabular}{' + 'l' * len(cols) + '}\n\\hline\n')
            f.write(' & '.join(tex(c) for c in cols) + ' \\\\\n\\hline\n')
            for _, r in table.iterrows():
                f.write(' & '.join(tex(r[c]) for c in cols) + ' \\\\\n')
            f.write('\\hline\n\\end{tabular}\n')

    def generate_detailed_report(self):
        path = os.path.join(self.results_dir, 'degradation_analysis_report.txt')
        t = f'{self.threshold:g}%'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"DEGRADATION ANALYSIS COMPREHENSIVE REPORT\n{RULE}\n")
            f.write(f"Threshold: {t}. Bootstrap draws: {self.n_boot}. Seed: {SEED}. Intervals are 95%.\n")
            for dataset, results in self.results.items():
                df = self.dataframes[dataset]
                f.write(f"\n\n{RULE}\nANALYSIS FOR DATASET: '{dataset}'\n{RULE}\n")
                f.write(f"  Data Summary:\n  Data points: {len(df)}\n"
                        f"  Time range: {df['Day'].min():.1f} to {df['Day'].max():.1f} days\n\n")
                f.write(f"DETAILED RESULTS BY BATCH:\n{'=' * 40}\n")
                for batch in self.batches[dataset]:
                    f.write(f"\n{batch}:\n{'_' * (len(batch) + 1)}\n")
                    rec = results.get(batch)
                    if not rec or not rec['models']:
                        f.write("    No valid models created for this batch.\n"); continue
                    f.write(f"    Points: {len(rec['x'])}. Ranked by {rec['criterion']}. Best: {rec['best']}.\n")
                    for name in sorted(rec['models'], key=lambda m: rec['models'][m]['delta']):
                        fit = rec['models'][name]
                        m = fit['metrics']
                        f.write(f"    {name:<12}: R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}, AICc={_fmt(m['aicc'], 2)}, "
                                f"BIC={m['bic']:.2f}, LOOCV RMSE={_fmt(fit['loocv'])}, weight={fit['weight']:.3f}, "
                                f"Days to {t} = {_fmt_days(fit['d50'])} {_fmt_ci(*_interval(fit['draws']))}\n")
                        f.write(f"        Parameters: {_param_text(fit)}\n")
                    avg = rec['avg']
                    f.write(f"    Model averaged days to {t}: {_fmt_days(avg['d50'])} {_fmt_ci(avg['lo'], avg['hi'])}\n")
                comp = self.comparisons.get(dataset)
                if comp is not None and len(comp):
                    f.write(f"\n\nBATCH COMPARISON AGAINST THE CONTROL (extra sum of squares F-test, Holm correction)\n{'=' * 40}\n")
                    f.write(comp.to_string(index=False) + "\n")
        print(f"Detailed report saved to: {path}")

    def run_complete_analysis(self):
        print("\n🚀 Starting complete degradation analysis workflow...")
        for info in self.datasets_config:
            self._analyse_dataset(info)  # errors reach main, so success prints only on success
        self.create_summary_tables()
        self.create_performance_heatmaps()
        self.generate_detailed_report()
        print(f"\n{RULE}\n✅ ANALYSIS COMPLETED SUCCESSFULLY!\n{RULE}")


def _ask_number(prompt, default, cast, valid):
    while True:
        text = input(prompt).strip()
        try:
            value = default if text == '' else cast(text)
            if value is not None and valid(value): return value
        except ValueError:
            pass
        print("Error: Invalid input. Please try again.")


def main():
    print(f"{RULE}\nDEGRADATION ANALYSIS & EXTRAPOLATION TOOL\n{RULE}")
    datasets_config = []
    num_datasets = _ask_number("Enter the number of datasets to analyze (1-10): ", None, int, lambda v: 1 <= v <= 10)

    for i in range(num_datasets):
        print(f"\nConfiguring Dataset {i + 1}")
        name = input("Enter a descriptive name for this dataset: ").strip()
        while True:
            path = input(f"Enter the full path to the Excel file for '{name}': ").strip().strip('"')
            if os.path.isfile(path):
                datasets_config.append({'name': name, 'path': path}); break
            print(f"❌ File not found at '{path}'. Please try again.")

    available = list(MODELS)
    all_choice = str(len(available) + 1)
    print(f"\n{RULE}\nMODEL SELECTION\n{RULE}\nPlease select the models you want to run:")
    for i, model in enumerate(available):
        print(f"  {i + 1}: {model}")
    print(f"  {all_choice}: All")

    while True:
        choice = input("Enter your choice(s) (e.g., '1', '2,4', or 'all'): ").lower().strip()
        if choice in ('all', all_choice):
            selected = available
            break
        try:
            indices = {int(i) - 1 for i in choice.split(',')}
        except ValueError:
            print("Error: Invalid input. Please enter numbers separated by commas, or 'all'."); continue
        if all(0 <= i < len(available) for i in indices):
            selected = [available[i] for i in sorted(indices)]
            break
        print(f"Error: One or more numbers are out of the valid range (1-{len(available)}).")

    threshold = _ask_number("Degradation threshold in percent (press Enter for 50): ", 50.0, float, lambda v: 0 < v < 100)
    n_boot = _ask_number("Bootstrap draws for intervals (Enter for 300, 0 to skip): ", 300, int, lambda v: v == 0 or v >= 50)

    print(f"\nModels selected for analysis: {', '.join(selected)}")
    try:
        DegradationAnalysis(datasets_config, selected, threshold, n_boot).run_complete_analysis()
        print("\n🎉 Analysis complete! Check the 'graphs' and 'results' folders.")
    except (OSError, ValueError) as e:
        print(f"\n❌ A configuration or data error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
