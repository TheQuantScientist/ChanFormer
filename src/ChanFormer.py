import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─── Configuration ───────────────────────────────────────────────────────────────

DATA_ROOT = "/home/nckh2/qa/ChanFormer/dataset/crypto"
DATA_ROOT_2 = "/home/nckh2/qa/ChanFormer/dataset/stock"

crypto_files = {
    'ATOMUSDT': f'{DATA_ROOT}/ATOMUSDT_1d_full.csv',
    'BCHUSDT': f'{DATA_ROOT}/BCHUSDT_1d_full.csv',
    'DOTUSDT': f'{DATA_ROOT}/DOTUSDT_1d_full.csv',
    'HBARUSDT': f'{DATA_ROOT}/HBARUSDT_1d_full.csv',
    'LTCUSDT': f'{DATA_ROOT}/LTCUSDT_1d_full.csv',
    'MATICUSDT': f'{DATA_ROOT}/MATICUSDT_1d_full.csv',
    'NEARUSDT': f'{DATA_ROOT}/NEARUSDT_1d_full.csv',
    'SHIBUSDT': f'{DATA_ROOT}/SHIBUSDT_1d_full.csv',
    'SUIUSDT': f'{DATA_ROOT}/SUIUSDT_1d_full.csv',
    'XLMUSDT': f'{DATA_ROOT}/XLMUSDT_1d_full.csv',
}

stock_files = {
    'AAPL': f'{DATA_ROOT_2}/AAPL_1d_full.csv',
    'AMZN': f'{DATA_ROOT_2}/AMZN_1d_full.csv',
    'AVGO': f'{DATA_ROOT_2}/AVGO_1d_full.csv',
    'BRK-B': f'{DATA_ROOT_2}/BRK-B_1d_full.csv',
    'GOOGL': f'{DATA_ROOT_2}/GOOGL_1d_full.csv',
    'META': f'{DATA_ROOT_2}/META_1d_full.csv',
    'MSFT': f'{DATA_ROOT_2}/MSFT_1d_full.csv',
    'NVDA': f'{DATA_ROOT_2}/NVDA_1d_full.csv',
    'TSLA': f'{DATA_ROOT_2}/TSLA_1d_full.csv',
    'TSM': f'{DATA_ROOT_2}/TSM_1d_full.csv',
}

# Training hyperparameters
SEQ_LEN       = 90
BATCH_SIZE    = 512
EPOCHS        = 6000
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20

# ─── Data Preparation ────────────────────────────────────────────────────────────

def load_and_prepare_multi_asset_data(file_paths):
    data = {}
    ASSETS = list(file_paths.keys())
    for asset, path in file_paths.items():
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        data[asset] = df

    df_all = None
    for asset in ASSETS:
        df = data[asset].rename(columns={
            'open':   f'{asset}_open',
            'high':   f'{asset}_high',
            'low':    f'{asset}_low',
            'close':  f'{asset}_close',
            'volume': f'{asset}_volume'
        })
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.join(df, how='outer')

    return df_all.ffill().bfill().sort_index()


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        shape = (1, 1, num_features)
        if affine:
            self.weight = nn.Parameter(torch.ones(shape))
            self.bias   = nn.Parameter(torch.zeros(shape))
        else:
            self.register_buffer('weight', torch.ones(shape))
            self.register_buffer('bias',   torch.zeros(shape))

    def norm(self, x):
        mean  = x.mean(dim=1, keepdim=True).detach()
        stdev = x.std(dim=1, keepdim=True).detach() + self.eps
        x_norm = (x - mean) / stdev
        return x_norm * self.weight + self.bias, mean, stdev

    def denorm(self, x, mean, stdev):
        x = (x - self.bias) / (self.weight + self.eps)
        return x * stdev + mean


class MultiAssetNoPatchTransformer(nn.Module):
    """
    Exact no-patching logic from ablation: sequence flattened to single token per channel.
    Transformer processes channels independently (seq_len=1 per channel → no cross-attention).
    Predictions made per-channel (no cross-channel averaging).
    """
    def __init__(self,
                 num_features,
                 seq_len=90,
                 d_model=384,
                 nhead=6,
                 num_layers=4,
                 dropout=0.10,
                 use_revin=True,
                 use_pos_embed=True,
                 use_channel_mixer=True):
        super().__init__()

        self.use_revin = use_revin
        self.use_pos_embed = use_pos_embed
        self.use_channel_mixer = use_channel_mixer

        self.revin = RevIN(num_features, affine=True) if use_revin else None

        self.seq_len = seq_len

        # Flatten time → project to d_model (one token per channel)
        self.patch_embed = nn.Linear(seq_len, d_model)

        # Positional embedding for the single token
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_channel_mixer:
            self.channel_mixer = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
        else:
            self.channel_mixer = nn.Identity()

        self.head = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (B, L, C)
        B, L, C = src.shape

        if self.use_revin:
            x, mean, stdev = self.revin.norm(src)
        else:
            x = src
            mean = stdev = None

        # Flatten time → (B*C, 1, d)
        x = x.permute(0, 2, 1).reshape(B * C, L)
        x = self.patch_embed(x.unsqueeze(1))  # (B*C, 1, d)

        if self.use_pos_embed:
            x = x + self.pos_embed

        x = self.encoder(x)  # (B*C, 1, d)

        # Reshape back: (B, 1, C, d) effectively
        P = x.size(1)  # P=1
        x = x.view(B, C, P, -1).permute(0, 2, 1, 3)  # (B, P, C, d)
        x = x.reshape(B * P, C, -1)  # (B, C, d) since P=1

        x = self.channel_mixer(x)

        x = x.view(B, P, C, -1).mean(dim=1)  # (B, C, d) — mean over P=1 (no-op)

        pred = self.head(x).squeeze(-1)  # (B, C)

        if self.use_revin:
            pred = pred.unsqueeze(1)  # (B, 1, C)
            pred = self.revin.denorm(pred, mean, stdev).squeeze(1)

        return pred


class OneStepAheadDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.from_numpy(x), torch.from_numpy(y)


# ─── Training & Evaluation Helpers ──────────────────────────────────────────────

def create_dataloaders(scaled_train, scaled_val, seq_len, batch_size):
    train_ds = OneStepAheadDataset(scaled_train, seq_len)
    val_ds   = OneStepAheadDataset(scaled_val,   seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs, device, save_path="model_no_patch_best.pth"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cosine   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8, verbose=True)

    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device).float()
            by = by.to(device).float()
            bx = bx + torch.randn_like(bx) * 0.0015

            pred = model(bx)
            loss = loss_fn(pred, by)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device).float()
                by = by.to(device).float()
                pred = model(bx)
                val_loss += loss_fn(pred, by).item()
                n_batches += 1

        val_loss /= n_batches if n_batches > 0 else float('inf')

        scheduler_cosine.step()
        scheduler_plateau.step(val_loss)

        torch.cuda.empty_cache()
        gc.collect()

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1:4d}] train: {train_loss/len(train_loader):.7f}  val: {val_loss:.7f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path))
    print(f"→ Loaded best weights from {save_path}")
    return model


def evaluate_one_step(model, full_scaled, seq_len, batch_size, device, scalers, columns, test_data, test_days, ASSETS):
    dataset = OneStepAheadDataset(full_scaled, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device).float()
            pred = model(bx)
            preds.append(pred.cpu().numpy())
            trues.append(by.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    start_idx = len(full_scaled) - seq_len - test_days
    if start_idx < 0:
        start_idx = 0
    preds_test = preds[start_idx : start_idx + test_days]
    trues_test = trues[start_idx : start_idx + test_days]

    err = preds_test - trues_test
    global_mse  = np.mean(err ** 2)
    global_rmse = math.sqrt(global_mse)
    global_mae  = np.mean(np.abs(err))

    results_global = {
        'test_mse_global':  global_mse,
        'test_rmse_global': global_rmse,
        'test_mae_global':  global_mae,
    }

    results_per_asset = []
    for asset in ASSETS:
        asset_cols = [f"{asset}_{v}" for v in ['open','high','low','close','volume']]
        idxs = [columns.index(c) for c in asset_cols]
        err_asset = preds_test[:, idxs] - trues_test[:, idxs]
        mse  = np.mean(err_asset ** 2)
        rmse = math.sqrt(mse)
        mae  = np.mean(np.abs(err_asset))
        results_per_asset.append({
            'asset': asset,
            'mse_scaled': mse,
            'rmse_scaled': rmse,
            'mae_scaled': mae,
        })

    return results_global, results_per_asset, preds_test, trues_test


# ─── Main Execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    scenarios = ['crypto', 'stock']

    for scenario in scenarios:
        if scenario == 'crypto':
            file_paths = crypto_files
        elif scenario == 'stock':
            file_paths = stock_files
        else:
            file_paths = {**crypto_files, **stock_files}

        ASSETS = list(file_paths.keys())
        prefix = f"{scenario}_"

        print(f"\n{'='*70}")
        print(f"TRAINING NO-PATCHING MODEL FOR {scenario.upper()}")
        print('='*70)

        print("Loading & preparing data...")
        df_all = load_and_prepare_multi_asset_data(file_paths)
        columns = df_all.columns.tolist()
        n_features = len(columns)

        # Split & scale
        test_data     = df_all.iloc[-TEST_DAYS:]
        pre_test_data = df_all.iloc[:-TEST_DAYS]
        val_size      = int(len(pre_test_data) * VAL_FRACTION)

        val_data   = pre_test_data.iloc[-val_size:]
        train_data = pre_test_data.iloc[:-val_size]

        scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

        def scale_df(df):
            return pd.DataFrame(
                np.hstack([scalers[col].transform(df[[col]]) for col in columns]),
                index=df.index, columns=columns
            )

        scaled_train = scale_df(train_data).values.astype(np.float32)
        scaled_val   = scale_df(val_data).values.astype(np.float32)
        scaled_test  = scale_df(test_data).values.astype(np.float32)
        full_scaled  = np.concatenate((scaled_train, scaled_val, scaled_test))

        model = MultiAssetNoPatchTransformer(
            num_features=n_features,
            seq_len=SEQ_LEN,
            d_model=384,
            nhead=6,
            num_layers=4,
            dropout=0.10,
            use_revin=True,
            use_pos_embed=True,
            use_channel_mixer=True,
        ).to(device)

        train_loader, val_loader = create_dataloaders(scaled_train, scaled_val, SEQ_LEN, BATCH_SIZE)

        model = train_model(model, train_loader, val_loader, EPOCHS, device,
                            save_path=f"{prefix}model_no_patch_best.pth")

        global_res, per_asset_res, preds_test_scaled, trues_test_scaled = evaluate_one_step(
            model, full_scaled, SEQ_LEN, BATCH_SIZE, device,
            scalers, columns, test_data, TEST_DAYS, ASSETS
        )

        print(f"\nFinal results (no patching) for {scenario}:")
        print(f"Global RMSE (scaled): {global_res['test_rmse_global']:.6f}")
        print(f"Global MAE  (scaled): {global_res['test_mae_global']:.6f}")

        # ─── Inverse transform predictions for saving ────────────────────────────
        pred_prices = np.zeros_like(preds_test_scaled)
        for i, col in enumerate(columns):
            pred_prices[:, i] = scalers[col].inverse_transform(
                preds_test_scaled[:, i].reshape(-1, 1)
            ).ravel()

        # Create DataFrame with dates from test_data
        pred_df = pd.DataFrame(
            pred_prices,
            index=test_data.index,
            columns=columns
        )

        # Combine true and predicted values
        forecast_df = test_data.add_suffix('_true').copy()
        for col in columns:
            forecast_df[f'{col}_pred'] = pred_df[col]

        # Save the forecast file
        forecast_path = f"{scenario}_forecasts_no_patch_full_ohlcv2.csv"
        forecast_df.to_csv(forecast_path)

        # ─── Save metrics ────────────────────────────────────────────────────────
        pd.DataFrame([global_res]).to_csv(f"{prefix}no_patch_global_result2.csv", index=False)
        pd.DataFrame(per_asset_res).to_csv(f"{prefix}no_patch_per_asset_results.csv2", index=False)

        print("\nResults saved to:")
        print(f"• {prefix}no_patch_global_result.csv")
        print(f"• {prefix}no_patch_per_asset_results.csv")
        print(f"• {forecast_path}")

    print("\nAll scenarios completed.")