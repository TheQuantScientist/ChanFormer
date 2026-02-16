import os
import sys

sys.path.append('.')

import argparse
import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features

# Import benchmark models
from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.Informer import Model as Informer

# Optional iTransformer
try:
    from models.iTransformer import Model as iTransformer
    HAS_iTRANSFORMER = True
except ImportError:
    HAS_iTRANSFORMER = False
    print("iTransformer module not found – skipping.")

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT_CRYPTO = "./dataset/crypto"
DATA_ROOT_STOCK  = "./dataset/stock"

CRYPTO_FILES = {
    'ATOMUSDT': f'{DATA_ROOT_CRYPTO}/ATOMUSDT_1d_full.csv',
    'BCHUSDT':  f'{DATA_ROOT_CRYPTO}/BCHUSDT_1d_full.csv',
    'DOTUSDT':  f'{DATA_ROOT_CRYPTO}/DOTUSDT_1d_full.csv',
    'HBARUSDT': f'{DATA_ROOT_CRYPTO}/HBARUSDT_1d_full.csv',
    'LTCUSDT':  f'{DATA_ROOT_CRYPTO}/LTCUSDT_1d_full.csv',
    'MATICUSDT':f'{DATA_ROOT_CRYPTO}/MATICUSDT_1d_full.csv',
    'NEARUSDT': f'{DATA_ROOT_CRYPTO}/NEARUSDT_1d_full.csv',
    'SHIBUSDT': f'{DATA_ROOT_CRYPTO}/SHIBUSDT_1d_full.csv',
    'SUIUSDT':  f'{DATA_ROOT_CRYPTO}/SUIUSDT_1d_full.csv',
    'XLMUSDT':  f'{DATA_ROOT_CRYPTO}/XLMUSDT_1d_full.csv',
}

STOCK_FILES = {
    'AAPL':   f'{DATA_ROOT_STOCK}/AAPL_1d_full.csv',
    'AMZN':   f'{DATA_ROOT_STOCK}/AMZN_1d_full.csv',
    'AVGO':   f'{DATA_ROOT_STOCK}/AVGO_1d_full.csv',
    'BRK-B':  f'{DATA_ROOT_STOCK}/BRK-B_1d_full.csv',
    'GOOGL':  f'{DATA_ROOT_STOCK}/GOOGL_1d_full.csv',
    'META':   f'{DATA_ROOT_STOCK}/META_1d_full.csv',
    'MSFT':   f'{DATA_ROOT_STOCK}/MSFT_1d_full.csv',
    'NVDA':   f'{DATA_ROOT_STOCK}/NVDA_1d_full.csv',
    'TSLA':   f'{DATA_ROOT_STOCK}/TSLA_1d_full.csv',
    'TSM':    f'{DATA_ROOT_STOCK}/TSM_1d_full.csv',
}

# Training settings (same as before)
SEQ_LEN      = 90
LABEL_LEN    = 45
PRED_LEN     = 1
EPOCHS       = 6000
BATCH_SIZE   = 256
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE     = 700
MIN_DELTA    = 2e-9
TEST_DAYS    = 365
VAL_FRACTION = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = nn.MSELoss()


def load_and_combine_multi_asset_data(file_paths, assets):
    """Load CSVs, prefix columns with asset name, outer join, ffill/bfill"""
    data = {}
    for asset, path in file_paths.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[asset] = df

    df_all = None
    for asset in assets:
        renamed = data[asset].rename(columns={
            'open':   f'{asset}_open',
            'high':   f'{asset}_high',
            'low':    f'{asset}_low',
            'close':  f'{asset}_close',
            'volume': f'{asset}_volume'
        })
        df_all = renamed if df_all is None else df_all.join(renamed, how='outer')

    return df_all.ffill().bfill().sort_index()


class TimeSeriesForecastDataset(Dataset):
    def __init__(self, data: np.ndarray, time_features: np.ndarray,
                 seq_len: int, label_len: int, pred_len: int = 1):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len

    def __len__(self):
        return len(self.data) - self.total_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = s_end + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.time_features[s_begin:s_end]
        seq_y_mark = self.time_features[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float()
        )


def get_benchmark_models(n_features: int):
    common = {
        'seq_len': SEQ_LEN,
        'label_len': LABEL_LEN,
        'pred_len': PRED_LEN,
        'enc_in': n_features,
        'dec_in': n_features,
        'c_out': n_features,
        'd_model': 384,
        'n_heads': 6,
        'e_layers': 4,
        'd_layers': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'activation': 'gelu',
        'embed': 'timeF',
        'freq': 'd',
    }

    models = [
        {
            'name': 'Autoformer',
            'class': Autoformer,
            'configs': {**common, 'task_name': 'long_term_forecast',
                        'factor': 3, 'moving_avg': 25}
        },
        {
            'name': 'FEDformer',
            'class': FEDformer,
            'configs': {**common, 'task_name': 'long_term_forecast',
                        'factor': 3, 'moving_avg': 25,
                        'version': 'Fourier', 'mode_select': 'random', 'modes': 32}
        },
        {
            'name': 'Informer',
            'class': Informer,
            'configs': {**common, 'task_name': 'long_term_forecast',
                        'factor': 5, 'distil': True, 'output_attention': False}
        }
    ]

    if HAS_iTRANSFORMER:
        models.append({
            'name': 'iTransformer',
            'class': iTransformer,
            'configs': {**common, 'task_name': 'long_term_forecast',
                        'e_layers': 3, 'factor': 3}
        })

    return models


def scale_and_split_data(df_all: pd.DataFrame):
    test_data    = df_all.iloc[-TEST_DAYS:]
    pre_test_df  = df_all.iloc[:-TEST_DAYS]
    val_size     = int(len(pre_test_df) * VAL_FRACTION)
    val_data     = pre_test_df.iloc[-val_size:]
    train_data   = pre_test_df.iloc[:-val_size]

    columns = df_all.columns.tolist()

    scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

    def apply_scaling(df):
        arr = np.hstack([scalers[col].transform(df[[col]]) for col in columns])
        return pd.DataFrame(arr, index=df.index, columns=columns)

    scaled_train = apply_scaling(train_data).values.astype(np.float32)
    scaled_val   = apply_scaling(val_data).values.astype(np.float32)
    scaled_test  = apply_scaling(test_data).values.astype(np.float32)

    return scaled_train, scaled_val, scaled_test, test_data, columns, scalers


def run_scenario(scenario: str, file_paths: dict, assets: list):
    print(f"\n{'='*80}")
    print(f"RUNNING BENCHMARK FOR {scenario.upper()} ({len(assets)} assets)")
    print('='*80)

    print("Loading and preparing data...")
    df_all = load_and_combine_multi_asset_data(file_paths, assets)

    # Time features (once)
    dates = df_all.index
    time_feat = time_features(dates, freq='d').T.astype(np.float32)

    scaled_train, scaled_val, scaled_test, test_data, columns, scalers = \
        scale_and_split_data(df_all)

    n_features = len(columns)

    pre_test_values = np.concatenate((scaled_train, scaled_val))
    pre_test_time   = time_feat[:len(pre_test_values)]
    test_time       = time_feat[-TEST_DAYS:]

    benchmark_models = get_benchmark_models(n_features)

    global_results = []
    per_asset_results = []
    overfit_results = []
    forecasts = {}

    prefix = f"{scenario}_"

    for bm in benchmark_models:
        model_name = bm['name']
        print(f"\n--- Training {model_name} ---")

        args = argparse.Namespace(**bm['configs'])
        model = bm['class'](args).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        train_ds = TimeSeriesForecastDataset(scaled_train, pre_test_time[:len(scaled_train)], SEQ_LEN, LABEL_LEN, PRED_LEN)
        val_ds   = TimeSeriesForecastDataset(scaled_val,   pre_test_time[len(scaled_train):],   SEQ_LEN, LABEL_LEN, PRED_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                     torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                    dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                         torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

                    out = model(x, x_mark, dec_inp, y_mark)
                    pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out
                    loss = LOSS_FN(pred, y[:, -PRED_LEN:, :])
                    val_loss += loss.item()
                    n_batches += 1

            val_loss /= n_batches if n_batches > 0 else float('inf')
            train_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0

            if (epoch + 1) % 10 == 0:
                print(f"[{epoch+1:4d}] train: {train_avg:.6f}  val: {val_loss:.6f}")

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ─── Evaluation on test set ───────────────────────────────────────
        full_values = np.concatenate((pre_test_values, scaled_test))
        full_time   = np.concatenate((pre_test_time, test_time))

        test_ds = TimeSeriesForecastDataset(full_values, full_time, SEQ_LEN, LABEL_LEN, PRED_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                     torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                preds_list.append(pred.cpu().numpy())
                trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

        preds = np.concatenate(preds_list, axis=0).reshape(-1, n_features)
        trues = np.concatenate(trues_list, axis=0).reshape(-1, n_features)

        start_idx = len(pre_test_values) - SEQ_LEN
        preds_test = preds[start_idx : start_idx + TEST_DAYS]
        trues_test = trues[start_idx : start_idx + TEST_DAYS]

        err = preds_test - trues_test
        global_mse  = np.mean(err ** 2)
        global_rmse = math.sqrt(global_mse)
        global_mae  = np.mean(np.abs(err))

        global_results.append({
            'model': model_name,
            'scenario': scenario,
            'test_mse_global': global_mse,
            'test_rmse_global': global_rmse,
            'test_mae_global': global_mae,
        })

        for asset in assets:
            asset_cols = [f"{asset}_{v}" for v in ['open','high','low','close','volume']]
            idxs = [columns.index(c) for c in asset_cols if c in columns]

            err_asset = preds_test[:, idxs] - trues_test[:, idxs]
            mse  = np.mean(err_asset ** 2)
            rmse = math.sqrt(mse)
            mae  = np.mean(np.abs(err_asset))

            per_asset_results.append({
                'model': model_name,
                'scenario': scenario,
                'asset': asset,
                'mse_scaled': mse,
                'rmse_scaled': rmse,
                'mae_scaled': mae,
            })

        print(f"{model_name} global RMSE (scaled): {global_rmse:.6f} | MAE: {global_mae:.6f}")

        # Overfit check
        train_loader_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

        train_preds_list, train_trues_list = [], []
        with torch.no_grad():
            for batch in train_loader_eval:
                x, y, x_mark, y_mark = [t.to(DEVICE) for t in batch]
                dec_inp = torch.cat([y[:, :LABEL_LEN, :],
                                     torch.zeros_like(y[:, -PRED_LEN:, :])], dim=1)

                out = model(x, x_mark, dec_inp, y_mark)
                pred = out[:, -PRED_LEN:, :] if out.shape[1] > PRED_LEN else out

                train_preds_list.append(pred.cpu().numpy())
                train_trues_list.append(y[:, -PRED_LEN:, :].cpu().numpy())

        train_preds = np.concatenate(train_preds_list, axis=0).reshape(-1, n_features)
        train_trues = np.concatenate(train_trues_list, axis=0).reshape(-1, n_features)

        train_mse = np.mean((train_preds - train_trues) ** 2)

        overfit_results.append({
            'model': model_name,
            'scenario': scenario,
            'train_mse_scaled': train_mse,
            'test_mse_scaled': global_mse,
            'overfit_gap_scaled': global_mse - train_mse
        })

        print(f"Train MSE: {train_mse:.6f} | Test MSE: {global_mse:.6f} | Gap: {global_mse - train_mse:.6f}")

        # Save inverse-scaled forecasts
        pred_prices = np.zeros_like(preds_test)
        for i, col in enumerate(columns):
            pred_prices[:, i] = scalers[col].inverse_transform(
                preds_test[:, i].reshape(-1, 1)
            ).ravel()

        forecasts[model_name] = {
            'dates': test_data.index,
            'true_df': test_data.copy(),
            'pred_df': pd.DataFrame(pred_prices, index=test_data.index, columns=columns)
        }

        torch.cuda.empty_cache()
        gc.collect()

    # ─── Save results for this scenario ──────────────────────────────────────────
    pd.DataFrame(global_results).to_csv(f'{prefix}one_step_transformer_global.csv', index=False)
    pd.DataFrame(per_asset_results).to_csv(f'{prefix}one_step_transformer_per_asset.csv', index=False)
    pd.DataFrame(overfit_results).to_csv(f'{prefix}overfit_transformer.csv', index=False)

    for model_name, data in forecasts.items():
        df_export = data['true_df'].add_suffix('_true').copy()
        for col in columns:
            df_export[f'{col}_one_step'] = data['pred_df'][col]
        df_export.to_csv(f'{prefix}forecasts_{model_name}_full_ohlcv.csv')

    print(f"\n{scenario.upper()} benchmark finished.")
    print(f"Saved files with prefix '{prefix}':")
    print(f"  • {prefix}one_step_transformer_global.csv")
    print(f"  • {prefix}one_step_transformer_per_asset.csv")
    print(f"  • {prefix}overfit_transformer.csv")
    print(f"  • {prefix}forecasts_{{model}}_full_ohlcv.csv")


# ─── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scenarios = [
        ("crypto", CRYPTO_FILES, list(CRYPTO_FILES.keys())),
        ("stock",  STOCK_FILES,  list(STOCK_FILES.keys())),
    ]

    for scenario_name, paths, asset_list in scenarios:
        run_scenario(scenario_name, paths, asset_list)

    print("\nAll benchmarks completed.")