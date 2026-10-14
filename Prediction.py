# Required packages: pandas, matplotlib, scikit-learn, openpyxl, scipy, numpy
# Install them with: pip install -r requirements.txt

import os
import traceback
import warnings

import matplotlib
matplotlib.use('Agg')  # figures are only written to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# Plot text: Times New Roman, black, sizes 16 (titles), 14 (axis labels), 12 (everything else)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],

    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14, 'axes.labelweight': 'bold',
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
})

THRESHOLD = 50.0   # degradation level (%) whose crossing time is reported
EPS = 1e-9         # floor that keeps log and power functions defined at x = 0
MAX_EXP = 50.0     # exponent clip so exp() cannot overflow
RULE = '=' * 60


# Model functions. The same function is used to fit, to predict and to plot, with no shifted x.
def _linear(x, m, c): return m * x + c
def _logarithmic(x, a, b, c): return a * np.log(np.maximum(x + b, EPS)) + c
def _power_law(x, a, b, c): return a * np.power(np.maximum(x, EPS), b) + c
def _exponential(x, a, b, c): return a * np.exp(np.clip(b * x, -MAX_EXP, MAX_EXP)) + c


# name: (function, equation for the plot, line colour)
MODELS = {
    'Linear': (_linear, 'y = m·x + c', 'blue'),
    'Logarithmic': (_logarithmic, 'y = a·ln(x + b) + c', 'green'),
    'Power Law': (_power_law, 'y = a·x^b + c', 'purple'),
    'Exponential': (_exponential, 'y = a·exp(b·x) + c', 'brown'),
}


def _fit_model(x, y, name):
    """Fit one model from several starting points. Returns (params, metrics) or (None, None)."""
    func = MODELS[name][0]
    span, rng = max(x.max() - x.min(), EPS), max(y.max() - y.min(), EPS)
    slope, y0 = (y[-1] - y[0]) / span, y[0]
    a_exp = rng / np.expm1(2)
    starts = {
        'Linear': [[slope, y0], [0.1, 0.0]],
        'Logarithmic': [[rng, 1.0, y.min()], [10.0, 1.0, 0.0]],
        'Power Law': [[slope, 1.0, y0], [1.0, 0.5, 0.0]],
        # first guess is anchored at the first x so b * x stays small when time does not start at 0
        'Exponential': [[a_exp * np.exp(-2.0 * x.min() / span), 2.0 / span, y0 - a_exp],
                        [1.0, 0.01, y0]],
    }[name]

    best_params, best_metrics, best_r2 = None, None, -np.inf
    for p0 in starts:
        try:
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore', OptimizeWarning)
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                pred = func(x, *popt)
        except (RuntimeError, ValueError):  # no convergence or non-finite residuals
            continue
        if not np.isfinite(pred).all():
            continue
        r2 = r2_score(y, pred)
        if r2 >= 0 and r2 > best_r2:
            mse = mean_squared_error(y, pred)
            best_params, best_r2 = popt, r2
            best_metrics = {'r2': r2, 'mse': mse, 'rmse': np.sqrt(mse)}
    return best_params, best_metrics


def _days_to_threshold(days, pred, threshold=THRESHOLD):
    """First day the curve reaches the threshold (linear interpolation), or NaN if it never does."""
    hit = pred >= threshold
    if not hit.any():
        return np.nan
    i = int(hit.argmax())
    if i == 0:
        return float(days[0])
    return float(np.interp(threshold, pred[i - 1:i + 1], days[i - 1:i + 1]))


def _fmt_days(info, nd=2):
    if info is None:
        return 'N/A'
    return 'Not Reached' if np.isnan(info['days_to_50']) else f"{info['days_to_50']:.{nd}f}"


def _fmt_metric(info, key):
    return 'N/A' if info is None else f"{info['metrics'][key]:.4f}"


def _safe(text, extra=''):
    return ''.join(c for c in text if c.isalnum() or c in extra)


class DegradationAnalysis:
    """Fits the selected models to every batch of every dataset, then plots, tabulates and reports."""

    def __init__(self, datasets_config, selected_models):
        if not datasets_config: raise ValueError("Dataset configuration cannot be empty.")
        if not selected_models: raise ValueError("A list of models to run must be provided.")

        self.datasets_config = datasets_config
        self.selected_models = selected_models
        self.batches = ['Standard Batch (0%)', 'Neem, 25%', 'Eucalyptus, 25%', 'Moringa, 25%']
        self.dataframes, self.all_models = {}, {}

        self.output_dir = os.path.dirname(os.path.abspath(datasets_config[0]['path']))
        self.graphs_dir = os.path.join(self.output_dir, 'graphs')
        self.results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"\n✅ All outputs will be saved to '{self.graphs_dir}' and '{self.results_dir}'")

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

        raw = pd.DataFrame({'Day': pd.to_numeric(df[tcol], errors='coerce') / (24 if 'hour' in tcol.lower() else 1)})
        for batch in self.batches:
            if batch in df.columns:
                raw[f'{batch}_degradation'] = pd.to_numeric(df[batch], errors='coerce').clip(lower=0)
        if raw.shape[1] == 1:
            raise ValueError(f"None of the expected batch columns were found in {base}.")

        # Rows are sorted by time and every reading is kept. Gaps inside a series are filled by
        # linear interpolation on the Day values (uneven spacing is respected); leading and
        # trailing gaps stay empty so no data is invented at the edges.
        df = raw.dropna(subset=['Day']).sort_values('Day', kind='stable').reset_index(drop=True)
        for col in df.columns[1:]:
            ok = df[col].notna()
            if ok.any():
                gap = ~ok & df['Day'].between(df.loc[ok, 'Day'].iloc[0], df.loc[ok, 'Day'].iloc[-1])
                df.loc[gap, col] = np.interp(df.loc[gap, 'Day'], df.loc[ok, 'Day'], df.loc[ok, col])
        if len(df) < 3:
            raise ValueError(f"Insufficient data points (<3) in {base}.")

        out = os.path.join(self.results_dir, f'{os.path.splitext(base)[0].lower()}_processed.csv')
        df.to_csv(out, index=False)
        print(f"Successfully processed {raw.shape[1] - 1} batches. Saved to {out}")
        return df

    def _create_models_for_dataset(self, info):
        name, path = info['name'], info['path']
        df = self._preprocess_data(path)
        self.dataframes[name] = df
        models = self.all_models.setdefault(name, {})
        grid = np.linspace(0, max(2000.0, 1.25 * df['Day'].max() + 5), 4000)  # day 2000 as in the original, longer only if the data needs it

        print(f"\n{RULE}\nCreating models for '{name}'\n{RULE}")
        for batch in self.batches:
            col = f'{batch}_degradation'
            if col not in df.columns: continue
            print(f"\nProcessing {batch}")
            sub = df[['Day', col]].dropna()  # each batch uses only its own valid rows
            x, y = sub['Day'].to_numpy(), sub[col].to_numpy()
            if len(x) < 3 or np.ptp(y) == 0:
                print("    Skipped: fewer than 3 points or no variation."); continue

            models[batch] = {}
            for model in self.selected_models:
                params, metrics = _fit_model(x, y, model)
                if params is None:
                    print(f"    Failed to fit {model} model."); continue
                with np.errstate(all='ignore'):  # a huge fitted exponent may overflow far out; the check below reports it
                    pred = MODELS[model][0](grid, *params)
                if not np.isfinite(pred).all():
                    print(f"    {model}: extrapolation is not finite, skipped."); continue
                d50 = _days_to_threshold(grid, pred)
                print(f"    {model}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
                models[batch][model] = {'params': params, 'metrics': metrics, 'days_to_50': d50}
                self._plot_fit(x, y, grid, pred, batch, name, model, metrics, d50)

            fitted = models[batch]
            if fitted:
                best = max(fitted, key=lambda m: fitted[m]['metrics']['r2'])
                print(f"  Best model for {batch}: {best} (R² = {fitted[best]['metrics']['r2']:.4f})")

    def _plot_fit(self, x, y, grid, pred, batch, dataset, model, metrics, d50):
        eq, color = MODELS[model][1:]
        last = x.max()
        limit = np.ceil(1.25 * last / 5) * 5
        if np.isfinite(d50): limit = max(limit, np.ceil(d50 / 5) * 5)  # always show the data and the crossing
        keep = grid <= limit
        g, p = grid[keep], pred[keep]
        k = np.searchsorted(g, last, side='right')

        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            ax.scatter(x, y, color='red', s=50, zorder=5, label='Actual Data')
            ax.plot(g[:k], p[:k], color=color, linewidth=2.5, label=f"{model} Model (R² = {metrics['r2']:.4f})")
            ax.plot(g[max(k - 1, 0):], p[max(k - 1, 0):], color=color, linewidth=2.5, linestyle='--')
            ax.axvline(last, color='gray', linestyle=':', linewidth=2, label=f'End of Actual Data ({last:.0f} days)')
            ax.axhline(THRESHOLD, color='black', linestyle='--', alpha=0.7, label='50% Degradation Threshold')
            if np.isfinite(d50):
                ax.axvline(d50, color='orange', linestyle=':', alpha=0.8)
                right = d50 > limit / 2  # put the label on the side of the line that has room
                ax.text(d50 + (-0.01 if right else 0.01) * limit, 45, f'{d50:.1f} days', fontweight='bold',
                        rotation=90, ha='right' if right else 'left')
            ax.set(title=f"{batch} | {dataset}\n{model} Model Extrapolation", xlabel='Time (Days)',
                   ylabel='Degradation (%)', ylim=(0, 105), xlim=(0, limit))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.legend(loc='best')
            ax.text(0.02, 0.98, f"Model: {eq}\nR² = {metrics['r2']:.4f}\nRMSE = {metrics['rmse']:.4f}",
                    transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            fig.tight_layout()
            name = f"{_safe(batch, ' _')}_{_safe(dataset)}_{model}_model.png"
            fig.savefig(os.path.join(self.graphs_dir, name), dpi=300, bbox_inches='tight')
        finally:
            plt.close(fig)
        print(f"    Plot saved: {name}")

    def create_summary_tables(self):
        print(f"\n{RULE}\nDEGRADATION ANALYSIS SUMMARY\n{RULE}")
        for dataset, models_data in self.all_models.items():
            print(f"\nSUMMARY FOR: '{dataset}'")
            for model in self.selected_models:
                print(f"\n{model.upper()} MODEL RESULTS")
                rows = []
                for batch in self.batches:
                    info = models_data.get(batch, {}).get(model)
                    rows.append({'Batch': batch, 'Days to 50%': _fmt_days(info, 4),
                                 'R²': _fmt_metric(info, 'r2'), 'RMSE': _fmt_metric(info, 'rmse')})
                table = pd.DataFrame(rows)
                print(table.to_string(index=False))
                table.to_csv(os.path.join(self.results_dir, f'summary_{_safe(dataset)}_{model}.csv'), index=False)
            print(f"Results for '{dataset}' saved to CSV files.")

    def create_performance_heatmaps(self):
        print(f"\n{RULE}\nCREATING PERFORMANCE HEATMAPS\n{RULE}")
        for dataset, models_data in self.all_models.items():
            shape = (len(self.selected_models), len(self.batches))
            r2, rmse = np.full(shape, np.nan), np.full(shape, np.nan)
            for i, model in enumerate(self.selected_models):
                for j, batch in enumerate(self.batches):
                    info = models_data.get(batch, {}).get(model)
                    if info: r2[i, j], rmse[i, j] = info['metrics']['r2'], info['metrics']['rmse']
            safe = _safe(dataset)
            self._plot_heatmap(r2, f'{dataset}\nR² Performance', f'heatmap_{safe}_R2.png')
            self._plot_heatmap(rmse, f'{dataset}\nRMSE Performance', f'heatmap_{safe}_RMSE.png', cmap='RdYlGn_r')

    def _plot_heatmap(self, data, title, filename, cmap='RdYlGn'):
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            if np.isnan(data).all():
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set(xticks=[], yticks=[], title=title)
            else:
                ax.imshow(data, cmap=cmap, aspect='auto')
                ax.set(xticks=np.arange(len(self.batches)), yticks=np.arange(len(self.selected_models)),
                       xticklabels=self.batches, yticklabels=self.selected_models, title=title,
                       xlabel='Batches', ylabel='Models')
                ax.xaxis.labelpad = ax.yaxis.labelpad = 15
                plt.setp(ax.get_xticklabels(), rotation=15, ha='right', rotation_mode='anchor')
                for (i, j), v in np.ndenumerate(data):
                    if not np.isnan(v): ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=16)
                ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)  # cell borders
                ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
                ax.tick_params(which='minor', bottom=False, left=False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.graphs_dir, filename), dpi=300)
        finally:
            plt.close(fig)
        print(f"Heatmap saved: {filename}")

    def generate_detailed_report(self):
        report_path = os.path.join(self.results_dir, 'degradation_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"DEGRADATION ANALYSIS COMPREHENSIVE REPORT\n{RULE}\n")
            for dataset, models in self.all_models.items():
                df = self.dataframes[dataset]
                f.write(f"\n\n{RULE}\nANALYSIS FOR DATASET: '{dataset}'\n{RULE}\n")
                f.write(f"  Data Summary:\n  Data points: {len(df)}\n"
                        f"  Time range: {df['Day'].min():.1f} - {df['Day'].max():.1f} days\n\n")
                f.write(f"DETAILED RESULTS BY BATCH:\n{'=' * 40}\n")
                for batch in self.batches:
                    f.write(f"\n{batch}:\n{'_' * (len(batch) + 1)}\n")
                    batch_models = models.get(batch, {})
                    if not batch_models:
                        f.write("    No valid models created for this batch.\n"); continue
                    for name in self.selected_models:
                        if name in batch_models:
                            data = batch_models[name]
                            f.write(f"    {name:<12}: R²={data['metrics']['r2']:.4f}, "
                                    f"RMSE={data['metrics']['rmse']:.4f}, Days to 50% = {_fmt_days(data)}\n")
        print(f"Detailed report saved to: {report_path}")

    def run_complete_analysis(self):
        print("\n🚀 Starting complete degradation analysis workflow...")
        for info in self.datasets_config:
            self._create_models_for_dataset(info)  # errors reach main, so success prints only on success
        self.create_summary_tables()
        self.create_performance_heatmaps()
        self.generate_detailed_report()
        print(f"\n{RULE}\n✅ ANALYSIS COMPLETED SUCCESSFULLY!\n{RULE}")


def main():
    print(f"{RULE}\nDEGRADATION ANALYSIS & EXTRAPOLATION TOOL\n{RULE}")
    datasets_config = []

    while True:
        try:
            num_datasets = int(input("Enter the number of datasets to analyze (1-10): "))
            if 1 <= num_datasets <= 10: break
            print("Error: Please enter a number between 1 and 10.")
        except ValueError:
            print("Error: Invalid input. Please enter a whole number.")

    for i in range(num_datasets):
        print(f"\nConfiguring Dataset {i + 1}")
        name = input("Enter a descriptive name for this dataset: ").strip()
        while True:
            path = input(f"Enter the full path to the Excel file for '{name}': ").strip().strip('"')
            if os.path.isfile(path):
                datasets_config.append({'name': name, 'path': path}); break
            print(f"❌ File not found at '{path}'. Please try again.")

    available_models = list(MODELS)
    all_choice = str(len(available_models) + 1)
    print(f"\n{RULE}\nMODEL SELECTION\n{RULE}\nPlease select the models you want to run:")
    for i, model in enumerate(available_models):
        print(f"  {i + 1}: {model}")
    print(f"  {all_choice}: All")

    while True:
        choice = input("Enter your choice(s) (e.g., '1', '2,4', or 'all'): ").lower().strip()
        if choice in ('all', all_choice):
            selected = available_models
            break
        try:
            indices = {int(i) - 1 for i in choice.split(',')}
        except ValueError:
            print("Error: Invalid input. Please enter numbers separated by commas, or 'all'."); continue
        if all(0 <= i < len(available_models) for i in indices):
            selected = [available_models[i] for i in sorted(indices)]
            break
        print(f"Error: One or more numbers are out of the valid range (1-{len(available_models)}).")

    print(f"\nModels selected for analysis: {', '.join(selected)}")
    try:
        DegradationAnalysis(datasets_config, selected).run_complete_analysis()
        print("\n🎉 Analysis complete! Check the 'graphs' and 'results' folders.")
    except (OSError, ValueError) as e:
        print(f"\n❌ A configuration or data error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
