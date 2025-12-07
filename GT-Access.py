#%% -*- coding: utf-8 -*-
#%%
import os
import math
import pandas as pd
import os, random, numpy as np, torch
import os, torch, numpy as np, time
import numpy as np
from typing import Dict, Tuple, Optional
import time

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, recall_score, roc_auc_score,
    average_precision_score
)

#%% improt self-defined functions and model
from utility import (
    append_result_csv, LastStepSupervisionDataset, PrefixCollator,load_userhistory_from_pt,
    append_train_log,
)

from models import AccessSeqModel,OnlyGraphHead,OnlyTransformerHead

#%% =============== Optional: Simple OnlyGraph / OnlyTransformer ===============

def get_vocab_sizes(train_base, val_base, test_base):
    u_max = max(
        max(s["user_id"] for s in train_base),
        max(s["user_id"] for s in val_base),
        max(s["user_id"] for s in test_base),
    ) + 1
    r_max = max(
        max(max(s["resources"]) for s in train_base),
        max(max(s["resources"]) for s in val_base),
        max(max(s["resources"]) for s in test_base),
    ) + 1
    return u_max, r_max


# =============== Loader / Evaluation ===============
def build_loader(base_ds, batch_size=128, max_len=64, shuffle=False, num_workers=0, generator=None):
    """
    Build DataLoader with fixed generator for reproducibility.
    """
    view = LastStepSupervisionDataset(base_ds)
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PrefixCollator(max_len=max_len),
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )

@torch.no_grad()
def scan_threshold_balacc(y_true: np.ndarray, y_score: np.ndarray,
                          lo=0.01, hi=0.99, steps=199) -> float:
    best_thr, best_score = 0.5, -1.0
    for t in np.linspace(lo, hi, steps):
        yp = (y_score >= t).astype(np.int32)
        ba = balanced_accuracy_score(y_true, yp)
        if ba > best_score:
            best_score, best_thr = ba, t
    return float(best_thr)

@torch.no_grad()
def evaluate_table_metrics(model: nn.Module, loader: DataLoader, device: str, threshold: float=0.5):
    model.eval(); model.to(device)
    ys, ps = [], []
    for batch in loader:
        batch = {k: v.to(device) for k,v in batch.items()}
        z = model(batch["user_ids"], batch["prefix_resources"], batch["prefix_actions"],
                  batch["attn_mask"], batch["last_resource"])
        p = torch.sigmoid(z)
        ys.append(batch["labels"].float().cpu()); ps.append(p.cpu())
    y_true = torch.cat(ys).numpy()
    y_score = torch.cat(ps).numpy()
    y_pred  = (y_score >= threshold).astype(np.int32)

    # --- evaluation metrics ---
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 balanced_accuracy_score, recall_score, roc_auc_score,
                                 average_precision_score)
    acc = accuracy_score(y_true, y_pred) * 100
    pre_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    pre_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "Acc(%)": acc,
        "Macro Pre(%)": pre_mac * 100,
        "Macro Rec(%)": rec_mac * 100,
        "Macro F1(%)":  f1_mac  * 100,
        "Min Pre(%)":   pre_cls[0] * 100,
        "Min Rec(%)":   rec_cls[0] * 100,
        "Min F1(%)":    f1_cls[0]  * 100,
        # Additional (threshold-independent) analyses
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC(pos)": average_precision_score(y_true, y_score),
        "PR-AUC(neg)": average_precision_score(1-y_true, 1-y_score),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
        "Recall_neg": recall_score(y_true, y_pred, pos_label=0),
    }
    return metrics, y_true, y_score

def print_latex_row(name: str, m: Dict[str, float]):
    fmt = lambda x: f"{x:.2f}"
    row = (f"{name} & "
           f"{fmt(m['Acc(%)'])} & "
           f"{fmt(m['Macro Pre(%)'])} & {fmt(m['Macro Rec(%)'])} & {fmt(m['Macro F1(%)'])} & "
           f"{fmt(m['Min Pre(%)'])} & {fmt(m['Min Rec(%)'])} & {fmt(m['Min F1(%)'])} \\\\")
    print(row)

# =============== Weighted BCE (Eq. 11) ===============

def compute_class_weights_from_train(train_base, max_len=64) -> Tuple[float, float]:
    """Return (w1, w0); normalized to 0.5/ratio for expected weight ≈1"""
    loader = build_loader(train_base, batch_size=2048, max_len=max_len, shuffle=False)
    ys = []
    for b in loader:
        ys.append(b["labels"])
    y = torch.cat(ys).float().numpy()
    pos = float(y.mean()); neg = 1.0 - pos
    # 防守
    eps = 1e-8
    w1 = 0.5 / max(eps, pos)
    w0 = 0.5 / max(eps, neg)
    print(f"[class weights] pos_ratio={pos:.6f} -> w1={w1:.6f} (for a=1), w0={w0:.6f} (for a=0)")
    return w1, w0

def weighted_bce_logits(logits, labels, w1: float, w0: float):
    """Implements Eq.(11), labels∈{0,1}"""
    labels = labels.float()
    # Per-sample weighting equivalence: weight = w1*y + w0*(1-y)
    weights = torch.where(labels > 0.5,
                          torch.as_tensor(w1, device=labels.device),
                          torch.as_tensor(w0, device=labels.device))
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
#%%
# ==========================================================
# Main function: train_one for both training with early stop and evaluation
# ==========================================================
def train_one(
    variant: str,
    train_base, val_base, test_base,
    user_embedding, resource_embedding,
    max_len: int = 64, d_model: Optional[int] = None,
    nhead: int = 4, layers: int = 2, dropout: float = 0.1,
    batch_size: int = 128, epochs: int = 20, lr: float = 1e-4,
    device: Optional[str] = None,
    fixed_threshold: float = 0.5,
    es_metric: str = "val_auprc",        # or "val_loss"
    log_csv: Optional[str] = None,       # path to save training logs
    patience: int = 10,                   # early-stopping patience
    seed: int = 2025,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(train_base, batch_size=batch_size, max_len=max_len, shuffle=True) #
    val_loader   = build_loader(val_base,   batch_size=batch_size, max_len=max_len, shuffle=False) #
    test_loader  = build_loader(test_base,  batch_size=batch_size, max_len=max_len, shuffle=False) #

    # ==========================================================
    # Build the model
    # ==========================================================
    if variant == "only_graph":
        model = OnlyGraphHead(user_embedding, resource_embedding, hidden=2*(user_embedding.shape[1]), dropout=dropout)
        eff_d_model = user_embedding.shape[1]
    elif variant == "only_transformer":
        eff_d_model = d_model or user_embedding.shape[1]
        num_users_data, num_res_data = get_vocab_sizes(train_base, val_base, test_base)
        model = OnlyTransformerHead(
            num_users=num_users_data, num_resources=num_res_data,
            d_model=eff_d_model, nhead=nhead, num_layers=layers,
            dropout=dropout, max_len_pos=max_len
        )

    else:  # ours (GT-Access)
        eff_d_model = d_model or user_embedding.shape[1]
        model = AccessSeqModel(
            user_embedding_matrix=user_embedding,
            resource_embedding_matrix=resource_embedding,
            d_act=max(8, eff_d_model//16), d_model=eff_d_model,
            nhead=nhead, num_layers=layers, dropout=dropout, max_len_pos=max_len
        )

    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # Usually use 3%~10%
    min_lr_ratio = 0.01  # The minimum is 1% of lr

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ==========================================================
    # Class weights (Eq. 11)
    # ==========================================================
    w1, w0 = compute_class_weights_from_train(train_base, max_len=max_len)

    # ==========================================================
    # Early Stopping initialization
    # ==========================================================
    best_state = None
    no_improve = 0
    # Set checkpoint path
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"best_{variant}_len{max_len}_lr{lr}_seed{seed}.pt" #_seed{seed}
    )

    if es_metric == "val_loss":
        best_score, mode, metric_name = float("inf"), "min", "val_loss"
    else:
        best_score, mode, metric_name = -float("inf"), "max", "val_auprc"


    # ==========================================================
    # Training loop
    # ==========================================================
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n_sum = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z = model(batch["user_ids"], batch["prefix_resources"], batch["prefix_actions"],
                      batch["attn_mask"], batch["last_resource"])
            loss = weighted_bce_logits(z, batch["labels"], w1=w1, w0=w0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step() #update larning rate
            loss_sum += float(loss) * batch["labels"].shape[0]
            n_sum += batch["labels"].shape[0]
        train_loss = loss_sum / max(1, n_sum)

        # ======================================================
        # Validation phase
        # ======================================================
        model.eval()
        with torch.no_grad():
            vy, vs = [], []
            for b in val_loader:
                b = {k: v.to(device) for k, v in b.items()}
                z = model(b["user_ids"], b["prefix_resources"], b["prefix_actions"],
                          b["attn_mask"], b["last_resource"])
                vy.append(b["labels"].float().cpu())
                vs.append(torch.sigmoid(z).cpu())
        vy = torch.cat(vy).numpy()
        vs = torch.cat(vs).numpy()

        # ---- Compute the early-stopping monitor metric ----
        if es_metric == "val_loss":
            val_metric = - ( vy*np.log(np.clip(vs,1e-8,1-1e-8))*w1 +
                             (1-vy)*np.log(np.clip(1-vs,1e-8,1-1e-8))*w0 ).mean()
        else:
            from sklearn.metrics import average_precision_score
            val_metric = average_precision_score(vy, vs)

        improved = (val_metric < best_score) if mode == "min" else (val_metric > best_score)
        if improved:
            best_score = val_metric
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # --- Save to disk  ---
            torch.save(
                {
                    "epoch": best_epoch,
                    "variant": variant,
                    "max_len": max_len,
                    "lr": lr,
                    "seed": seed,
                    "best_metric": best_score,
                    "model_state": best_state,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path} (epoch={ep}, {es_metric}={val_metric:.4f})")

            no_improve = 0
        else:
            no_improve += 1

        print(f"[{variant:>15}] epoch {ep:03d}  train_loss={train_loss:.4f}  "
              f"{metric_name}={val_metric:.4f}  best={best_score:.4f}  no_improve={no_improve}")

        # ---- Write logs ----
        if log_csv is not None:
            append_train_log(log_csv, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "variant": variant,
                "epoch": ep,
                "train_loss": float(train_loss),
                "val_metric": float(val_metric),
                "metric_name": metric_name,
                "best_score": float(best_score),
                "no_improve": int(no_improve),
                "lr": float(lr),
                "max_len": int(max_len),
                "d_model": int(eff_d_model),
                "nhead": int(nhead),
                "layers": int(layers),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "w1": float(w1),
                "w0": float(w0),
                "fixed_threshold": float(fixed_threshold),
                "patience": int(patience)
            })

        # ---- Early stop  ----
        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {ep} (no improvement {patience}).")
            break

    # ==========================================================
    # Load best model & final evaluation
    # ==========================================================
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, y_true, y_score = evaluate_table_metrics(
        model, test_loader, device=device, threshold=fixed_threshold
    )

    print(f"\n=== Save results: ({variant}, thr={fixed_threshold:.2f}) ===")
    print_latex_row("GT-Access" if variant=="ours" else variant.replace("_","-"), metrics)

    # Metadata summary
    meta = {
        "variant": variant,
        "seed": seed,
        "max_len": max_len,
        "d_model": eff_d_model,
        "nhead": nhead,
        "layers": layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "fixed_threshold": fixed_threshold,
        "earlystop_metric": es_metric,
        "w1": w1, "w0": w0,
        "patience": patience
    }
    return metrics, fixed_threshold, meta


#%% ==================== Usage Example====================
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # 1) 1) Load GNN embeddings (index equals mapped ID)
    emb = torch.load('./data/None_BS_1024_HC_16_Node_embeddings.pt')
    User_embedding = emb['User_embedding'].detach().cpu().numpy()
    Resource_embedding = emb['Resource_embedding'].detach().cpu().numpy()

    # 2) Prepare data for Transformer: train_base / val_base / test_base
    train_base = load_userhistory_from_pt("./data/train_dataset.pt")
    val_base = load_userhistory_from_pt("./data/val_dataset.pt")
    test_base = load_userhistory_from_pt("./data/test_dataset.pt")

    # 3) Run three variants (choose as needed) for ablastion study and save results
    results_csv = "./data/runs_results.csv"
    TABLE_COLS = [
        "Acc(%)", "Macro Pre(%)", "Macro Rec(%)", "Macro F1(%)",
        "Min Pre(%)", "Min Rec(%)", "Min F1(%)",
        # Optional: store these too for easier follow-up analyses
        "ROC-AUC", "PR-AUC(pos)", "PR-AUC(neg)", "BalAcc", "Recall_neg"
    ]
    Path("logs").mkdir(exist_ok=True)

    # set the same max_len for ablation study
    max_len = 8
    runs=10
    for i in range(runs):
        # # 1) Ours
        m_ours, thr_ours, meta_ours = train_one(
            "ours", train_base, val_base, test_base,
            User_embedding, Resource_embedding,
            max_len=max_len, epochs=1000, lr=1e-4, device=device,
            fixed_threshold=0.5, es_metric="val_auprc",
            log_csv="./logs/train_{variant}_len{max_len}_seed{seed}.csv",  # <<< log file
            patience=10
        )
        append_result_csv(results_csv, meta_ours, m_ours,TABLE_COLS)

        # 2) Only-Graph
        m_g, thr_g, meta_g = train_one(
            "only_graph", train_base, val_base, test_base,
            User_embedding, Resource_embedding,
            max_len=max_len, epochs=1000, lr=1e-4, device=device,
            fixed_threshold=0.5, es_metric="val_auprc",
            log_csv="./logs/train_{variant}_len{max_len}_seed{seed}.csv",
            patience=10
        )
        append_result_csv(results_csv, meta_g, m_g,TABLE_COLS)

        # 3) Only-Transformer（match d_model with the graph side）
        m_t, thr_t, meta_t = train_one(
            "only_transformer", train_base, val_base, test_base,
            User_embedding, Resource_embedding,
            d_model=User_embedding.shape[1],
            max_len=max_len, epochs=1000, lr=1e-4, device=device,
            fixed_threshold=0.5, es_metric="val_auprc",
            log_csv="./logs/train_{variant}_len{max_len}_seed{seed}.csv",
            patience=10
        )
        append_result_csv(results_csv, meta_t, m_t,TABLE_COLS)

    for i in range(runs):
        # # hyperparameter adjustment (max_lens)
        max_lens=[16,32,64]
        for max_len in max_lens:
            m_ours, thr_ours, meta_ours = train_one(
                "ours", train_base, val_base, test_base,
                User_embedding, Resource_embedding,
                max_len=max_len, epochs=1000, lr=1e-4, device=device,
                fixed_threshold=0.5, es_metric="val_auprc",
                log_csv="./logs/train_{variant}_len{max_len}_seed{seed}.csv",  # <<<  log file
                patience=10
            )
            append_result_csv(results_csv, meta_ours, m_ours,TABLE_COLS)

