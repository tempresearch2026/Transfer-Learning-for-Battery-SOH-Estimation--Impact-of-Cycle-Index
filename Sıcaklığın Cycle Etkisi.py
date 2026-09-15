# ============================================================================
# BATTERY SOH ESTIMATION WITH TRANSFER LEARNING
# Cycle-Index Inclusion Effect Under Temperature-Induced Domain Shift
# FINAL VERSION v2 - Imputation and Subset Validation Fixed
# ============================================================================

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, wilcoxon
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SOURCE_DATA_PATH = os.path.join("data", "source_data.csv")
    TARGET_DATA_PATH = os.path.join("data", "target_data.csv")

    if not os.path.exists(SOURCE_DATA_PATH):
        SOURCE_DATA_PATH = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\25 degree\kaynak_birlesik_veri.csv"
        TARGET_DATA_PATH = r"C:\Users\LENOVO\Desktop\Transfer Learning Project\Veriler\cleaned_dataset\organized_dataset\4 degree\hedef_B0056_discharge_standardized1.csv"

    OUTPUT_DIR = "results_final_v2"
    RANDOM_STATE = 42

    TEMPORAL_TRAIN_RATIO = 0.60
    TEMPORAL_VAL_RATIO = 0.20
    TEMPORAL_TEST_RATIO = 0.20

    ENSEMBLE_N_MODELS = 5
    ENSEMBLE_MAX_DEPTH_VARIANTS = [4, 5, 6, 5, 4]
    ENSEMBLE_LR_VARIANTS = [0.05, 0.06, 0.07, 0.08, 0.09]

    PROGRESSIVE_ITERATIONS = 3
    PROGRESSIVE_LR_FACTORS = [1.0, 0.5, 0.3]

    SELECTIVE_TAU = 0.7

    TARGET_SUBSET_CYCLE_FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.0]

    N_SEEDS = 10
    SEEDS = [42, 123, 456, 789, 101112, 2024, 31415, 27182, 16180, 14142]

    FEATURES_BASE = ['terminal_voltage', 'terminal_current', 'temperature']
    FEATURES_WITH_CYCLE = ['terminal_voltage', 'terminal_current', 'temperature', 'cycle']


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def calculate_cycle_level_metrics(y_true, y_pred, cycle_values):
    unique_cycles = np.unique(cycle_values)
    cycle_rmse = {}
    cycle_mae = {}
    for c in unique_cycles:
        mask = cycle_values == c
        if mask.sum() > 0:
            cycle_rmse[c] = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            cycle_mae[c] = mean_absolute_error(y_true[mask], y_pred[mask])
    return cycle_rmse, cycle_mae


def mmd_rbf(x, y, sigma=1.0):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = pairwise_distances(x, x, metric='sqeuclidean')
    yy = pairwise_distances(y, y, metric='sqeuclidean')
    xy = pairwise_distances(x, y, metric='sqeuclidean')
    k_xx = np.exp(-xx / (2 * sigma ** 2))
    k_yy = np.exp(-yy / (2 * sigma ** 2))
    k_xy = np.exp(-xy / (2 * sigma ** 2))
    return np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)


def calculate_feature_wise_domain_gap(X_source, X_target, feature_names, random_state=42):
    # FIX: Fixed random seed for reproducibility
    rng = np.random.default_rng(random_state)
    results = {}
    for i, feat in enumerate(feature_names):
        src = X_source[:, i]
        tgt = X_target[:, i]

        n_sub = min(1000, len(src), len(tgt))
        src_sub = rng.choice(src, n_sub, replace=False)
        tgt_sub = rng.choice(tgt, n_sub, replace=False)

        results[feat] = {
            'wasserstein': wasserstein_distance(src, tgt),
            'ks_statistic': ks_2samp(src, tgt)[0],
            'ks_p_value': ks_2samp(src, tgt)[1],
            'mmd': mmd_rbf(src_sub, tgt_sub),
            'cohens_d': cohens_d(src, tgt)
        }
    return results


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std


def paired_cohens_dz(x, y):
    diff = x - y
    if np.std(diff, ddof=1) == 0:
        return 0
    return np.mean(diff) / np.std(diff, ddof=1)


def rank_biserial_from_wilcoxon(x, y):
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0
    ranks = stats.rankdata(np.abs(diff))
    pos_ranks = np.sum(ranks[diff > 0])
    neg_ranks = np.sum(ranks[diff < 0])
    total = pos_ranks + neg_ranks
    return (pos_ranks - neg_ranks) / total if total > 0 else 0


def holm_correction(p_values, alpha=0.05):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[i] = min(1.0, p * (n - i))
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result, result < alpha


def format_p(p):
    if p < 1e-10:
        return "p < 1e-10"
    elif p < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p:.4f}"


# ============================================================================
# DATA LOADING - FIX: NO IMPUTATION HERE (done after split)
# ============================================================================

def load_data_no_imputation(path, feature_list):
    """
    Load data WITHOUT imputation. Imputation is done after temporal split
    using training-set medians only.
    """
    df = pd.read_csv(path)

    if 'SOH' not in df.columns:
        soh_cols = [c for c in df.columns if 'soh' in c.lower()]
        if soh_cols:
            df.rename(columns={soh_cols[0]: 'SOH'}, inplace=True)

    df = df.select_dtypes(include=[np.number])
    df = df.dropna(subset=['SOH'])

    if df['SOH'].max() <= 1.5:
        df['SOH'] = df['SOH'] * 100
    elif df['SOH'].max() > 150:
        df['SOH'] = df['SOH'] / 100

    X = df[feature_list].copy()
    y = df['SOH'].copy()
    cycle = df['cycle'].copy() if 'cycle' in df.columns else None

    return X, y, cycle, df


def impute_train_only(X_train, X_val, X_test):
    """Impute missing values using medians computed ONLY from training data."""
    medians = X_train.median()
    return X_train.fillna(medians), X_val.fillna(medians), X_test.fillna(medians)


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def temporal_split_by_cycle(X, y, cycle_values, train_ratio=0.60, val_ratio=0.20):
    unique_cycles = np.sort(np.unique(cycle_values))
    n_cycles = len(unique_cycles)

    train_idx = int(n_cycles * train_ratio)
    val_idx = int(n_cycles * (train_ratio + val_ratio))

    train_cycles = unique_cycles[:train_idx]
    val_cycles = unique_cycles[train_idx:val_idx]
    test_cycles = unique_cycles[val_idx:]

    train_mask = np.isin(cycle_values, train_cycles)
    val_mask = np.isin(cycle_values, val_cycles)
    test_mask = np.isin(cycle_values, test_cycles)

    return {
        'X_train': X[train_mask].copy(),
        'X_val': X[val_mask].copy(),
        'X_test': X[test_mask].copy(),
        'y_train': y[train_mask].copy(),
        'y_val': y[val_mask].copy(),
        'y_test': y[test_mask].copy(),
        'train_cycles': train_cycles,
        'val_cycles': val_cycles,
        'test_cycles': test_cycles
    }


# ============================================================================
# TRANSFER STRATEGIES
# ============================================================================

class TransferStrategies:
    def __init__(self, base_model, base_booster):
        self.base_model = base_model
        self.base_booster = base_booster

    def scratch(self, X_train, y_train, X_test, seed=42):
        model = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def base(self, X_test):
        return self.base_model.predict(X_test)

    def progressive(self, X_train, y_train, X_test, outer_seed=42):
        current_model = None
        predictions = []
        for i in range(Config.PROGRESSIVE_ITERATIONS):
            lr_factor = Config.PROGRESSIVE_LR_FACTORS[i]
            learning_rate = 0.1 * lr_factor
            model = XGBRegressor(
                n_estimators=20, learning_rate=learning_rate, max_depth=5,
                random_state=outer_seed + i, n_jobs=-1
            )
            if i == 0:
                model.fit(X_train, y_train, xgb_model=self.base_booster, verbose=False)
            else:
                model.fit(X_train, y_train, xgb_model=current_model.get_booster(), verbose=False)
            current_model = model
            predictions.append(current_model.predict(X_test))

        weights = np.linspace(0.1, 0.9, Config.PROGRESSIVE_ITERATIONS)
        weights = weights / weights.sum()
        y_final = np.zeros_like(predictions[0])
        for pred, w in zip(predictions, weights):
            y_final += pred * w
        return y_final

    def validation_weighted_ensemble(self, X_train, y_train, X_val, y_val, X_test, outer_seed=42):
        predictions_val = []
        predictions_test = []

        for i in range(Config.ENSEMBLE_N_MODELS):
            member_seed = outer_seed * 1000 + i
            model = XGBRegressor(
                n_estimators=30,
                learning_rate=Config.ENSEMBLE_LR_VARIANTS[i],
                max_depth=Config.ENSEMBLE_MAX_DEPTH_VARIANTS[i],
                subsample=0.7 + (i * 0.05),
                random_state=member_seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train, xgb_model=self.base_booster, verbose=False)
            predictions_val.append(model.predict(X_val))
            predictions_test.append(model.predict(X_test))

        preds_val = np.column_stack(predictions_val)
        preds_test = np.column_stack(predictions_test)

        std_val = np.std(preds_val, axis=1, ddof=1)
        std_test = np.std(preds_test, axis=1, ddof=1)

        member_val_rmse = []
        for i in range(Config.ENSEMBLE_N_MODELS):
            member_val_rmse.append(np.sqrt(mean_squared_error(y_val, preds_val[:, i])))
        member_val_rmse = np.array(member_val_rmse)

        weights = 1.0 / (member_val_rmse + 1e-6)
        weights = weights / weights.sum()

        weighted_mean_test = np.sum(preds_test * weights[np.newaxis, :], axis=1)

        uncertainty_stats = {
            'method': 'Validation-Weighted Ensemble with Epistemic Uncertainty Estimation',
            'mean_epistemic_uncertainty_test': float(np.mean(std_test)),
            'max_epistemic_uncertainty_test': float(np.max(std_test)),
            'mean_epistemic_uncertainty_val': float(np.mean(std_val)),
            'member_val_rmse': member_val_rmse.tolist(),
            'member_weights': weights.tolist()
        }
        return weighted_mean_test, uncertainty_stats

    def selective_agreement(self, X_train, y_train, X_test, outer_seed=42):
        temp_model = XGBRegressor(
            n_estimators=30, max_depth=4, learning_rate=0.1,
            random_state=outer_seed, n_jobs=-1
        )
        temp_model.fit(X_train, y_train)
        base_pred_train = self.base_model.predict(X_train)
        target_pred_train = temp_model.predict(X_train)
        agreement = np.abs(base_pred_train - target_pred_train)
        threshold = np.percentile(agreement, Config.SELECTIVE_TAU * 100)
        selected_idx = agreement <= threshold
        if selected_idx.sum() < 10:
            selected_idx = np.ones(len(X_train), dtype=bool)
        model = XGBRegressor(
            n_estimators=30, max_depth=5, learning_rate=0.05,
            random_state=outer_seed, n_jobs=-1
        )
        model.fit(X_train[selected_idx], y_train[selected_idx],
                  xgb_model=self.base_booster, verbose=False)
        return model.predict(X_test)

    def ridge_calibration(self, X_train, y_train, X_test):
        base_pred_train = self.base_model.predict(X_train).reshape(-1, 1)
        base_pred_test = self.base_model.predict(X_test).reshape(-1, 1)
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(base_pred_train, y_train)
        return meta_learner.predict(base_pred_test)


# ============================================================================
# SUBSET VALIDATION SPLIT - FIX: CYCLE-LEVEL
# ============================================================================

def cycle_level_subset_split(X_sub, y_sub, cycle_sub, val_ratio=0.20):
    """
    Split subset chronologically at CYCLE level (not sample level).
    Ensures no cycle appears in both train and validation.
    """
    unique_cycles = np.sort(np.unique(cycle_sub))
    n_cycles = len(unique_cycles)

    if n_cycles < 2:
        # Fallback: use all for training
        return X_sub, y_sub, None, None

    val_start = max(1, int(n_cycles * (1 - val_ratio)))
    val_cycles = unique_cycles[val_start:]
    train_cycles = unique_cycles[:val_start]

    train_mask = np.isin(cycle_sub, train_cycles)
    val_mask = np.isin(cycle_sub, val_cycles)

    return (X_sub[train_mask], y_sub[train_mask],
            X_sub[val_mask], y_sub[val_mask])


# ============================================================================
# SCENARIO RUNNER
# ============================================================================

def run_scenario(scenario_name, feature_list, source_path, target_path, output_dir):
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Features: {feature_list}")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. SOURCE DATA + TEMPORAL SPLIT + TRAIN-ONLY IMPUTATION
    # ------------------------------------------------------------------------
    print("\n[1] Source data loading (25C)...")
    X_source, y_source, cycle_source, df_source = load_data_no_imputation(source_path, feature_list)

    source_split = temporal_split_by_cycle(
        X_source, y_source, cycle_source.values,
        Config.TEMPORAL_TRAIN_RATIO, Config.TEMPORAL_VAL_RATIO
    )

    # FIX: Imputation after split using train-only medians
    X_train_s, X_val_s, X_test_s = impute_train_only(
        source_split['X_train'], source_split['X_val'], source_split['X_test']
    )
    y_train_s = source_split['y_train']
    y_val_s = source_split['y_val']
    y_test_s = source_split['y_test']

    print(f"    Source train cycles: {source_split['train_cycles'][0]}-{source_split['train_cycles'][-1]}")
    print(f"    Source test cycles: {source_split['test_cycles'][0]}-{source_split['test_cycles'][-1]}")

    scaler = RobustScaler()
    X_train_s_scaled = scaler.fit_transform(X_train_s)
    X_test_s_scaled = scaler.transform(X_test_s)

    source_model = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=Config.RANDOM_STATE, n_jobs=-1
    )
    source_model.fit(X_train_s_scaled, y_train_s)

    source_metrics = calculate_metrics(y_test_s, source_model.predict(X_test_s_scaled))
    print(f"    Source Test RMSE: {source_metrics['RMSE']:.4f}%")

    base_booster = source_model.get_booster()

    importance_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': source_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # ------------------------------------------------------------------------
    # 2. TARGET DATA + TEMPORAL SPLIT + TRAIN-ONLY IMPUTATION
    # ------------------------------------------------------------------------
    print("\n[2] Target data loading (4C)...")
    X_target, y_target, cycle_target, df_target = load_data_no_imputation(target_path, feature_list)

    target_split = temporal_split_by_cycle(
        X_target, y_target, cycle_target.values,
        Config.TEMPORAL_TRAIN_RATIO, Config.TEMPORAL_VAL_RATIO
    )

    # FIX: Imputation after split using train-only medians
    X_train, X_val, X_test = impute_train_only(
        target_split['X_train'], target_split['X_val'], target_split['X_test']
    )
    y_train = target_split['y_train']
    y_val = target_split['y_val']
    y_test = target_split['y_test']
    train_cycles = target_split['train_cycles']
    val_cycles = target_split['val_cycles']
    test_cycles = target_split['test_cycles']

    cycle_test = cycle_target.values[np.isin(cycle_target.values, test_cycles)]
    cycle_train = cycle_target.values[np.isin(cycle_target.values, train_cycles)]

    print(f"    Target train cycles: {train_cycles[0]}-{train_cycles[-1]} ({len(train_cycles)})")
    print(f"    Target test cycles: {test_cycles[0]}-{test_cycles[-1]} ({len(test_cycles)})")
    print(f"    Target test SOH range: {y_test.min():.2f} - {y_test.max():.2f}%")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------------
    # 3. DOMAIN GAP ANALYSIS
    # ------------------------------------------------------------------------
    print("\n[3] Domain gap analysis...")

    source_soh_train = y_train_s.values
    target_soh_train = y_train.values

    soh_gap = {
        'wasserstein': wasserstein_distance(source_soh_train, target_soh_train),
        'ks_statistic': ks_2samp(source_soh_train, target_soh_train)[0],
        'ks_p_value': ks_2samp(source_soh_train, target_soh_train)[1],
    }

    feature_gap = calculate_feature_wise_domain_gap(
        X_train_s_scaled, X_train_scaled, feature_list
    )

    print(f"    SOH Distribution Gap: Wasserstein={soh_gap['wasserstein']:.4f}, KS={soh_gap['ks_statistic']:.4f}")

    with open(os.path.join(output_dir, 'domain_gap_metrics.json'), 'w') as f:
        json.dump({'soh_distribution_gap_train': soh_gap, 'feature_wise_gap': feature_gap}, f, indent=2)

    # ------------------------------------------------------------------------
    # 4. TRANSFER STRATEGIES
    # ------------------------------------------------------------------------
    print("\n[4] Transfer strategies...")
    transfer = TransferStrategies(source_model, base_booster)

    all_metrics = {}
    all_predictions = {}

    t0 = time.time()
    y_pred = transfer.scratch(X_train_scaled, y_train.values, X_test_scaled)
    all_metrics['Scratch'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Scratch'] = y_pred

    t0 = time.time()
    y_pred = transfer.base(X_test_scaled)
    all_metrics['Base'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Base'] = y_pred

    t0 = time.time()
    y_pred = transfer.progressive(X_train_scaled, y_train.values, X_test_scaled)
    all_metrics['Progressive'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Progressive'] = y_pred

    t0 = time.time()
    y_pred, unc_stats = transfer.validation_weighted_ensemble(
        X_train_scaled, y_train.values, X_val_scaled, y_val.values, X_test_scaled
    )
    all_metrics['Validation-Weighted Ensemble'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Validation-Weighted Ensemble'] = y_pred

    t0 = time.time()
    y_pred = transfer.selective_agreement(X_train_scaled, y_train.values, X_test_scaled)
    all_metrics['Selective'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Selective'] = y_pred

    t0 = time.time()
    y_pred = transfer.ridge_calibration(X_train_scaled, y_train.values, X_test_scaled)
    all_metrics['Ridge-Calibration'] = {**calculate_metrics(y_test.values, y_pred), 'Time': time.time() - t0}
    all_predictions['Ridge-Calibration'] = y_pred

    for name, m in all_metrics.items():
        print(f"    {name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    pd.DataFrame(all_metrics).T.to_csv(os.path.join(output_dir, 'all_models_metrics.csv'))

    # ------------------------------------------------------------------------
    # 5. STATISTICAL TESTS (cycle-level with explicit cycle ID matching)
    # ------------------------------------------------------------------------
    print("\n[5] Statistical tests (cycle-level, Holm-corrected)...")

    cycle_rmse_dict = {}
    for name, pred in all_predictions.items():
        cycle_rmse, _ = calculate_cycle_level_metrics(y_test.values, pred, cycle_test)
        cycle_rmse_dict[name] = cycle_rmse

    stat_results = []
    p_values_raw = []

    for name in all_predictions.keys():
        if name == 'Scratch':
            continue

        # FIX: explicit cycle-ID-based matching
        common_cycles = sorted(set(cycle_rmse_dict['Scratch'].keys()) & set(cycle_rmse_dict[name].keys()))
        if len(common_cycles) < 5:
            continue

        scratch_arr = np.array([cycle_rmse_dict['Scratch'][c] for c in common_cycles])
        model_arr = np.array([cycle_rmse_dict[name][c] for c in common_cycles])

        stat, p = wilcoxon(scratch_arr, model_arr)
        dz = paired_cohens_dz(scratch_arr, model_arr)
        rb = rank_biserial_from_wilcoxon(scratch_arr, model_arr)

        p_values_raw.append(p)
        stat_results.append({
            'Strategy': name,
            'Wilcoxon_p_raw': p,
            'Cohens_dz': dz,
            'Rank_Biserial_r': rb,
            'N_Cycles_Compared': len(common_cycles)
        })

    if len(p_values_raw) > 0:
        adjusted_p, significant = holm_correction(p_values_raw)
        for i, res in enumerate(stat_results):
            res['Wilcoxon_p_holm'] = adjusted_p[i]
            res['Significant_after_Holm'] = bool(significant[i])
            res['Wilcoxon_p_formatted'] = format_p(adjusted_p[i])

    stat_df = pd.DataFrame(stat_results)
    stat_df.to_csv(os.path.join(output_dir, 'statistical_tests.csv'), index=False)
    print(stat_df.to_string(index=False))

    # ------------------------------------------------------------------------
    # 6. SUBSET ANALYSIS (FIX: cycle-level validation split)
    # ------------------------------------------------------------------------
    print("\n[6] Subset analysis (cycle-based, cycle-level validation)...")

    subset_results = []
    unique_train_cycles = np.sort(np.unique(cycle_train))
    n_train_cycles = len(unique_train_cycles)

    for frac in Config.TARGET_SUBSET_CYCLE_FRACTIONS:
        n_cycles = max(1, int(n_train_cycles * frac))
        subset_cycles = unique_train_cycles[:n_cycles]

        subset_mask = np.isin(cycle_train, subset_cycles)
        X_sub = X_train_scaled[subset_mask]
        y_sub = y_train.values[subset_mask]
        cycle_sub = cycle_train[subset_mask]

        if len(X_sub) < 10:
            print(f"    Fraction {frac*100:.0f}%: skipped (n<10)")
            continue

        scratch_rmses = []
        progressive_rmses = []
        uncertainty_rmses = []

        for seed in Config.SEEDS:
            # FIX: cycle-level split for ensemble validation
            X_sub_train, y_sub_train, X_sub_val, y_sub_val = cycle_level_subset_split(
                X_sub, y_sub, cycle_sub, val_ratio=0.20
            )

            if X_sub_val is None or len(X_sub_val) < 2:
                # Fallback: use a small random subset for validation
                n_val = max(2, int(len(X_sub) * 0.2))
                X_sub_train = X_sub[:-n_val]
                y_sub_train = y_sub[:-n_val]
                X_sub_val = X_sub[-n_val:]
                y_sub_val = y_sub[-n_val:]

            y_scratch = transfer.scratch(X_sub, y_sub, X_test_scaled, seed=seed)
            scratch_rmses.append(np.sqrt(mean_squared_error(y_test, y_scratch)))

            y_prog = transfer.progressive(X_sub, y_sub, X_test_scaled, outer_seed=seed)
            progressive_rmses.append(np.sqrt(mean_squared_error(y_test, y_prog)))

            y_unc, _ = transfer.validation_weighted_ensemble(
                X_sub_train, y_sub_train, X_sub_val, y_sub_val,
                X_test_scaled, outer_seed=seed
            )
            uncertainty_rmses.append(np.sqrt(mean_squared_error(y_test, y_unc)))

        subset_results.append({
            'Cycle_Fraction': frac,
            'N_Cycles': n_cycles,
            'N_Samples': len(X_sub),
            'Scratch_RMSE_Mean': np.mean(scratch_rmses),
            'Scratch_RMSE_Std': np.std(scratch_rmses, ddof=1),
            'Progressive_RMSE_Mean': np.mean(progressive_rmses),
            'Progressive_RMSE_Std': np.std(progressive_rmses, ddof=1),
            'Uncertainty_RMSE_Mean': np.mean(uncertainty_rmses),
            'Uncertainty_RMSE_Std': np.std(uncertainty_rmses, ddof=1)
        })
        print(f"    Fraction {frac*100:.0f}% ({n_cycles} cycles, {len(X_sub)} samples): "
              f"Scratch={np.mean(scratch_rmses):.4f}±{np.std(scratch_rmses, ddof=1):.4f}, "
              f"Uncertainty={np.mean(uncertainty_rmses):.4f}±{np.std(uncertainty_rmses, ddof=1):.4f}")

    subset_df = pd.DataFrame(subset_results)
    subset_df.to_csv(os.path.join(output_dir, 'subset_analysis.csv'), index=False)

    # ------------------------------------------------------------------------
    # 7. VISUALIZATIONS
    # ------------------------------------------------------------------------
    print("\n[7] Visualizations...")

    n_pca = min(2000, len(X_train_s_scaled), len(X_train_scaled))
    combined = np.vstack([X_train_s_scaled[:n_pca], X_train_scaled[:n_pca]])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pca_result[:n_pca, 0], pca_result[:n_pca, 1], c='blue', alpha=0.3, s=5, label='Source (25°C)')
    ax.scatter(pca_result[n_pca:, 0], pca_result[n_pca:, 1], c='red', alpha=0.3, s=5, label='Target (4°C)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(f'PCA: Source vs Target Training Distribution - {scenario_name}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pca_domain_shift.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(all_metrics.keys())
    rmses = [all_metrics[n]['RMSE'] for n in names]
    colors = ['#E74C3C' if n == 'Scratch' else '#3498DB' for n in names]
    bars = ax.bar(names, rmses, color=colors, alpha=0.8)
    ax.set_ylabel('RMSE (% SOH)')
    ax.set_title(f'Transfer Strategy Comparison - {scenario_name}')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'transfer_strategies_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()

    if len(subset_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(subset_df['N_Samples'], subset_df['Scratch_RMSE_Mean'],
                    yerr=subset_df['Scratch_RMSE_Std'], fmt='o-', label='Scratch', capsize=5, linewidth=2)
        ax.errorbar(subset_df['N_Samples'], subset_df['Progressive_RMSE_Mean'],
                    yerr=subset_df['Progressive_RMSE_Std'], fmt='s-', label='Progressive', capsize=5, linewidth=2)
        ax.errorbar(subset_df['N_Samples'], subset_df['Uncertainty_RMSE_Mean'],
                    yerr=subset_df['Uncertainty_RMSE_Std'], fmt='^-', label='Validation-Weighted Ensemble',
                    capsize=5, linewidth=2)
        ax.set_xlabel('Number of Target Training Samples'); ax.set_ylabel('RMSE (% SOH)')
        ax.set_title(f'Data Efficiency Analysis - {scenario_name}')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'subset_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Importance'); ax.set_title(f'Feature Importance - {scenario_name}')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------------
    # 8. FINAL SUMMARY
    # ------------------------------------------------------------------------
    best_strategy = min(all_metrics.keys(), key=lambda x: all_metrics[x]['RMSE'])

    summary = {
        'scenario': scenario_name,
        'features': feature_list,
        'source_rmse': source_metrics['RMSE'],
        'best_strategy': best_strategy,
        'best_rmse': all_metrics[best_strategy]['RMSE'],
        'scratch_rmse': all_metrics['Scratch']['RMSE'],
        'base_rmse': all_metrics['Base']['RMSE'],
        'soh_distribution_gap_train': soh_gap,
        'feature_wise_gap': feature_gap,
        'all_metrics': all_metrics,
        'statistical_tests': stat_results,
        'subset_analysis': subset_results
    }

    with open(os.path.join(output_dir, 'final_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n    Best: {best_strategy} (RMSE={all_metrics[best_strategy]['RMSE']:.4f})")

    return {
        'scenario': scenario_name,
        'metrics': all_metrics,
        'predictions': all_predictions,
        'y_test': y_test.values,
        'cycle_test': cycle_test,
        'summary': summary,
        'subset_df': subset_df,
        'stat_results': stat_results
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("BATTERY SOH TRANSFER LEARNING - FINAL v2")
    print("="*80)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    results_with_cycle = run_scenario(
        scenario_name="Cycle_Included",
        feature_list=Config.FEATURES_WITH_CYCLE,
        source_path=Config.SOURCE_DATA_PATH,
        target_path=Config.TARGET_DATA_PATH,
        output_dir=os.path.join(Config.OUTPUT_DIR, "cycle_included")
    )

    results_without_cycle = run_scenario(
        scenario_name="Cycle_Excluded",
        feature_list=Config.FEATURES_BASE,
        source_path=Config.SOURCE_DATA_PATH,
        target_path=Config.TARGET_DATA_PATH,
        output_dir=os.path.join(Config.OUTPUT_DIR, "cycle_excluded")
    )

    print("\n" + "="*80)
    print("COMPARISON: CYCLE-INDEX INCLUSION EFFECT")
    print("="*80)

    comp_dir = os.path.join(Config.OUTPUT_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    comparison_data = []
    for strategy in results_with_cycle['metrics'].keys():
        comparison_data.append({
            'Strategy': strategy,
            'RMSE_Cycle_Included': results_with_cycle['metrics'][strategy]['RMSE'],
            'RMSE_Cycle_Excluded': results_without_cycle['metrics'][strategy]['RMSE'],
            'MAE_Cycle_Included': results_with_cycle['metrics'][strategy]['MAE'],
            'MAE_Cycle_Excluded': results_without_cycle['metrics'][strategy]['MAE']
        })

    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(os.path.join(comp_dir, 'cycle_index_inclusion_comparison.csv'), index=False)
    print(comp_df.to_string(index=False))

    # Statistical test
    stat_results_comp = []
    p_values_raw_comp = []

    for strategy in results_with_cycle['metrics'].keys():
        cycle_rmse_with, _ = calculate_cycle_level_metrics(
            results_with_cycle['y_test'], results_with_cycle['predictions'][strategy],
            results_with_cycle['cycle_test']
        )
        cycle_rmse_without, _ = calculate_cycle_level_metrics(
            results_without_cycle['y_test'], results_without_cycle['predictions'][strategy],
            results_without_cycle['cycle_test']
        )

        # Explicit cycle-ID-based matching
        common_cycles = sorted(set(cycle_rmse_with.keys()) & set(cycle_rmse_without.keys()))
        if len(common_cycles) < 5:
            continue

        arr_with = np.array([cycle_rmse_with[c] for c in common_cycles])
        arr_without = np.array([cycle_rmse_without[c] for c in common_cycles])

        stat, p = wilcoxon(arr_with, arr_without)
        dz = paired_cohens_dz(arr_without, arr_with)
        rb = rank_biserial_from_wilcoxon(arr_without, arr_with)

        p_values_raw_comp.append(p)
        stat_results_comp.append({
            'Strategy': strategy,
            'Wilcoxon_p_raw': p,
            'Cohens_dz': dz,
            'Rank_Biserial_r': rb,
            'N_Cycles': len(common_cycles)
        })

    if len(p_values_raw_comp) > 0:
        adjusted_p, significant = holm_correction(p_values_raw_comp)
        for i, res in enumerate(stat_results_comp):
            res['Wilcoxon_p_holm'] = adjusted_p[i]
            res['Significant_after_Holm'] = bool(significant[i])
            res['Wilcoxon_p_formatted'] = format_p(adjusted_p[i])

    pd.DataFrame(stat_results_comp).to_csv(
        os.path.join(comp_dir, 'cycle_index_inclusion_statistical_test.csv'), index=False
    )
    print(pd.DataFrame(stat_results_comp).to_string(index=False))

    # Comparison figure
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df['RMSE_Cycle_Included'], width, label='Cycle Included', color='#3498DB', alpha=0.8)
    ax.bar(x + width/2, comp_df['RMSE_Cycle_Excluded'], width, label='Cycle Excluded', color='#E74C3C', alpha=0.8)
    ax.set_ylabel('RMSE (% SOH)')
    ax.set_title('Effect of Cycle-Index Inclusion on Transfer Learning Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Strategy'], rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'cycle_index_inclusion_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    final_comparison = {
        'cycle_included': {
            'best_strategy': results_with_cycle['summary']['best_strategy'],
            'best_rmse': results_with_cycle['summary']['best_rmse'],
            'scratch_rmse': results_with_cycle['summary']['scratch_rmse'],
            'base_rmse': results_with_cycle['summary']['base_rmse']
        },
        'cycle_excluded': {
            'best_strategy': results_without_cycle['summary']['best_strategy'],
            'best_rmse': results_without_cycle['summary']['best_rmse'],
            'scratch_rmse': results_without_cycle['summary']['scratch_rmse'],
            'base_rmse': results_without_cycle['summary']['base_rmse']
        }
    }

    with open(os.path.join(comp_dir, 'final_comparison.json'), 'w') as f:
        json.dump(final_comparison, f, indent=2, default=str)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()