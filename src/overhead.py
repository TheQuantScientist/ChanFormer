import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import gc
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─── Resource & Energy Tracking ─────────────────────────────────────────────────
from codecarbon import EmissionsTracker
import pynvml
import psutil

try:
    pynvml.nvmlInit()
    HAVE_NVIDIA = pynvml.nvmlDeviceGetCount() > 0
except:
    HAVE_NVIDIA = False
    print("No NVIDIA GPU detected — GPU metrics unavailable.")

def get_gpu_util_and_power():
    if not HAVE_NVIDIA:
        return 0.0, 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        return util, power
    except:
        return 0.0, 0.0

def log_resources(phase="", epoch=None):
    cpu_pct = psutil.cpu_percent(interval=None)
    ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    gpu_util, gpu_power = get_gpu_util_and_power()
    msg = f"[{phase}]"
    if epoch is not None:
        msg += f" Epoch {epoch+1:4d} |"
    msg += f" CPU: {cpu_pct:5.1f}% | RAM: {ram_gb:5.1f} GB"
    if HAVE_NVIDIA:
        msg += f" | GPU util: {gpu_util:5.1f}% | GPU power: {gpu_power:6.1f} W"
    print(msg)

# ─── FLOPs / MACs ────────────────────────────────────────────────────────────────
try:
    from thop import profile, clever_format
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False
    print("thop not installed — FLOPs/MACs skipped. Install with: pip install thop")

# ─── Configuration ───────────────────────────────────────────────────────────────
DATA_ROOT_CRYPTO = "/home/nckh2/qa/ChanFormer/dataset/crypto"
DATA_ROOT_STOCK  = "/home/nckh2/qa/ChanFormer/dataset/stock"

crypto_files = {
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

stock_files = {
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

scenarios = {
    'crypto': crypto_files,
    'stock':  stock_files
}

# Hyperparameters
SEQ_LEN       = 90
BATCH_SIZE    = 256
EPOCHS        = 7000
LR            = 8e-5
WEIGHT_DECAY  = 5e-6
PATIENCE      = 700
MIN_DELTA     = 2e-9
TEST_DAYS     = 365
VAL_FRACTION  = 0.20

# Benchmark settings
BENCH_WARMUP  = 20
BENCH_RUNS    = 50
BENCH_BATCH   = 512

# ─── Model ───────────────────────────────────────────────────────────────────────
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

        self.patch_embed = nn.Linear(seq_len, d_model)

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
        B, L, C = src.shape

        if self.use_revin:
            x, mean, stdev = self.revin.norm(src)
        else:
            x = src
            mean = stdev = None

        x = x.permute(0, 2, 1).reshape(B * C, L)
        x = self.patch_embed(x.unsqueeze(1))

        if self.use_pos_embed:
            x = x + self.pos_embed

        x = self.encoder(x)

        x = x.view(B, C, -1)
        x = self.channel_mixer(x)
        pred = self.head(x).squeeze(-1)

        if self.use_revin:
            pred = pred.unsqueeze(1)
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

# ─── Helpers ─────────────────────────────────────────────────────────────────────
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

    return df_all.ffill().bfill().sort_index(), ASSETS

def create_dataloaders(scaled_train, scaled_val, seq_len, batch_size):
    train_ds = OneStepAheadDataset(scaled_train, seq_len)
    val_ds   = OneStepAheadDataset(scaled_val, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

def benchmark_inference(model, device, seq_len, num_features, batch_size=BENCH_BATCH):
    model.eval()
    dummy = torch.randn(batch_size, seq_len, num_features, device=device)

    print("Benchmark warmup...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(BENCH_WARMUP):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("done.")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)

    times_ms = []
    with torch.no_grad():
        for _ in range(BENCH_RUNS):
            if device.type == 'cuda':
                start_ev.record()
            _ = model(dummy)
            if device.type == 'cuda':
                end_ev.record()
                torch.cuda.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    mean_ms  = times_ms.mean()
    p95_ms   = np.percentile(times_ms, 95)
    throughput = (batch_size * BENCH_RUNS) / (times_ms.sum() / 1000)

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs_str = flops_str = "N/A"
    if HAVE_THOP:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs_str, _ = clever_format([macs, num_params], "%.3f")
        gflops = float(macs) * 2 / 1e9
        flops_str = f"{gflops:.2f} GFLOPs (fwd)"

    print("\n" + "═"*90)
    print(f" BENCHMARK ({device.type.upper()}) — batch={batch_size}  seq={seq_len}  features={num_features}")
    print("─"*90)
    print(f" Throughput          : {throughput:10.1f} samples/sec")
    print(f" Latency p95         : {p95_ms:8.1f} ms          ← key for real-time-ish use")
    print(f" Peak GPU memory     : {peak_gb:6.2f} GB")
    print(f" MACs / GFLOPs (fwd) : {macs_str}   ≈ {flops_str}")
    print("═"*90 + "\n")

    return {
        'throughput_samples_sec': throughput,
        'latency_p95_ms': p95_ms,
        'peak_memory_gb': peak_gb,
        'macs': macs_str,
        'gflops_fwd': flops_str,
        'mean_latency_ms': mean_ms,
        'params': num_params
    }

# ─── Main Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    for scenario_name, file_paths in scenarios.items():
        print(f"\n{'═'*100}")
        print(f"   NO-PATCH TRANSFORMER — {scenario_name.upper()}")
        print(f"{'═'*100}\n")

        print("Loading & preparing data...")
        df_all, ASSETS = load_and_prepare_multi_asset_data(file_paths)
        columns = df_all.columns.tolist()
        n_features = len(columns)

        test_data     = df_all.iloc[-TEST_DAYS:]
        pre_test_data = df_all.iloc[:-TEST_DAYS]
        val_size      = int(len(pre_test_data) * VAL_FRACTION)
        val_data      = pre_test_data.iloc[-val_size:]
        train_data    = pre_test_data.iloc[:-val_size]

        scalers = {col: MinMaxScaler().fit(train_data[[col]]) for col in columns}

        def scale_df(df):
            arr = np.hstack([scalers[col].transform(df[[col]]) for col in columns])
            return pd.DataFrame(arr, index=df.index, columns=columns).values.astype(np.float32)

        scaled_train = scale_df(train_data)
        scaled_val   = scale_df(val_data)
        scaled_test  = scale_df(test_data)

        train_loader, val_loader = create_dataloaders(scaled_train, scaled_val, SEQ_LEN, BATCH_SIZE)

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

        # ── Training with energy tracking ────────────────────────────────────────
        print(f"Training {scenario_name} model...")
        tracker_train = EmissionsTracker(
            project_name=f"NoPatch_Train_{scenario_name}",
            output_dir="/home/nckh2/qa/ChanFormer/src",
            measure_power_secs=10,
            log_level="error"
        )
        tracker_train.start()

        start_train = time.time()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler_cosine  = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8, verbose=True)

        loss_fn = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx = bx.to(device).float()
                by = by.to(device).float()
                bx += torch.randn_like(bx) * 0.0015

                pred = model(bx)
                loss = loss_fn(pred, by)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                train_loss += loss.item()

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
                print(f"[{epoch+1:4d}] train: {train_loss/len(train_loader):.7f}  val: {val_loss:.7f}  "
                      f"lr: {optimizer.param_groups[0]['lr']:.2e}")
                log_resources(f"{scenario_name.upper()} TRAIN+VAL", epoch)

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{scenario_name}_nopatch_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        train_runtime_s = time.time() - start_train
        tracker_train.stop()

        print(f"\nTraining summary ({scenario_name}):")
        print(f"  Runtime           : {train_runtime_s:.1f} s")
        print(f"  Energy consumed   : {tracker_train.final_emissions_data.energy_consumed:.4f} kWh")
        print(f"  CO₂ equivalent    : {tracker_train.final_emissions_data.emissions:.4f} kg")

        # Load best model
        model.load_state_dict(torch.load(f"{scenario_name}_nopatch_best.pth", weights_only=True))
        print("Loaded best weights.\n")

        # ── Inference benchmark ──────────────────────────────────────────────────
        bench_result = benchmark_inference(model, device, SEQ_LEN, n_features, batch_size=BENCH_BATCH)

        # ── Final test evaluation + energy tracking ──────────────────────────────
        full_scaled = np.concatenate([scaled_train, scaled_val, scaled_test])
        test_ds = OneStepAheadDataset(full_scaled, SEQ_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        tracker_inf = EmissionsTracker(
            project_name=f"NoPatch_Infer_{scenario_name}",
            output_dir="/home/nckh2/qa/ChanFormer/src",
            measure_power_secs=5,
            log_level="error"
        )
        tracker_inf.start()

        start_inf = time.time()
        preds, trues = [], []
        gpu_utils_inf, gpu_powers_inf = [], []

        model.eval()
        with torch.no_grad():
            for i, (bx, by) in enumerate(test_loader):
                bx = bx.to(device).float()
                pred = model(bx)
                preds.append(pred.cpu().numpy())
                trues.append(by.numpy())

                if HAVE_NVIDIA:
                    u, p = get_gpu_util_and_power()
                    gpu_utils_inf.append(u)
                    gpu_powers_inf.append(p)

                if i % 50 == 0 and i > 0:
                    log_resources(f"{scenario_name.upper()} INF batch {i*BATCH_SIZE:,}")

        inf_runtime_s = time.time() - start_inf
        tracker_inf.stop()

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        start_idx = max(0, len(full_scaled) - SEQ_LEN - TEST_DAYS)
        preds_test = preds[start_idx : start_idx + TEST_DAYS]
        trues_test = trues[start_idx : start_idx + TEST_DAYS]

        err = preds_test - trues_test
        global_mse  = np.mean(err ** 2)
        global_rmse = math.sqrt(global_mse)
        global_mae  = np.mean(np.abs(err))

        print(f"\nEvaluation summary ({scenario_name}):")
        print(f"  Global RMSE (scaled): {global_rmse:.6f}")
        print(f"  Global MAE  (scaled): {global_mae:.6f}")
        print(f"  Inference runtime   : {inf_runtime_s:.1f} s")
        print(f"  ~ per sample        : {inf_runtime_s / len(test_ds):.4f} s")

        if HAVE_NVIDIA and gpu_utils_inf:
            print(f"  Avg GPU util (inf)  : {np.mean(gpu_utils_inf):.1f}%")
            print(f"  Avg GPU power (inf) : {np.mean(gpu_powers_inf):.1f} W")

        print(f"  Energy consumed     : {tracker_inf.final_emissions_data.energy_consumed:.4f} kWh")
        print(f"  CO₂ equivalent      : {tracker_inf.final_emissions_data.emissions:.4f} kg")

        # ── Save results ─────────────────────────────────────────────────────────
        summary_row = {
            'scenario': scenario_name,
            'test_rmse_global': global_rmse,
            'test_mae_global': global_mae,
            'throughput_samples_sec': bench_result['throughput_samples_sec'],
            'latency_p95_ms': bench_result['latency_p95_ms'],
            'peak_memory_gb': bench_result['peak_memory_gb'],
            'gflops_fwd': bench_result['gflops_fwd'],
            'train_runtime_s': train_runtime_s,
            'inf_runtime_s': inf_runtime_s,
            'train_energy_kwh': tracker_train.final_emissions_data.energy_consumed,
            'train_co2_kg': tracker_train.final_emissions_data.emissions,
            'inf_energy_kwh': tracker_inf.final_emissions_data.energy_consumed,
            'inf_co2_kg': tracker_inf.final_emissions_data.emissions,
        }

        pd.DataFrame([summary_row]).to_csv(f"{scenario_name}_nopatch_full_summary.csv", index=False)
        print(f"Full summary saved → {scenario_name}_nopatch_full_summary.csv\n")

    print("\nAll scenarios completed.")
    print("Check ./emissions_nopatch/ for detailed emissions logs.")