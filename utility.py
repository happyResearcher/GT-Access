import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from typing import Optional
from torch.utils.data import Dataset
import pandas as pd
import random
import os
from torch.utils.data import Subset
from typing import List, Dict
import numpy as np
import csv
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class UserHistoryDataset(Dataset):
    def __init__(self, history_df: pd.DataFrame):
        """
        Build a dataset grouped by PERSON_ID.
        For each user, save all the history sequence of resource ids (TARGET_NAME)
        and the corresponding labels (ACTION).

        Args:
            history_df: DataFrame with columns ['ACTION', 'TARGET_NAME', 'PERSON_ID']
        """
        self.user_histories = {}
        grouped = history_df.groupby('PERSON_ID')
        for user_id, group in grouped:
            # Sort by row order (if time order not explicit)
            seq_resources = group['TARGET_NAME'].tolist()
            seq_actions = group['ACTION'].tolist()
            self.user_histories[user_id] = {
                'resources': seq_resources,
                'actions': seq_actions
            }

        # Save all user ids for indexing
        self.user_ids = list(self.user_histories.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        history = self.user_histories[user_id]
        return {
            'user_id': user_id,
            'resources': history['resources'],
            'actions': history['actions']
        }

def append_train_log(csv_path: str, row: dict):
    """
    Append one training log entry to a CSV file.
    If the file does not exist, the header will be written automatically.
    Example of `row`:
      {"time": "...", "variant": "ours", "epoch": 12, "train_loss": 0.43,
       "val_metric": 0.592, "metric_name": "val_loss", "best_score": 0.571,
       "no_improve": 3, "lr": 1e-4}
    """
    p = Path(csv_path)
    is_new = not p.exists()
    with p.open("a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def append_result_csv(csv_path: str, meta: Dict, metrics: Dict,TABLE_COLS:List):
    """Append one experiment’s metadata and metrics to a CSV file。"""
    csv_path = Path(csv_path)
    is_new = not csv_path.exists()

    # Construct the row: write experiment settings first, followed by metric values
    row = {
        **meta,
        **{k: float(metrics[k]) for k in TABLE_COLS if k in metrics}
    }

    # Header = keys from meta + metric columns
    fieldnames = list(meta.keys()) + TABLE_COLS

    with csv_path.open("a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"[saved] append to {csv_path}")

def load_userhistory_from_pt(path: str):
    samples = torch.load(path)
    rows = []
    for s in samples:
        uid = s["user_id"]
        for r, a in zip(s["resources"], s["actions"]):
            rows.append({"PERSON_ID": uid, "TARGET_NAME": r, "ACTION": a})
    return UserHistoryDataset(pd.DataFrame(rows))


# ========== Collate Function: Pad variable-length prefixes and generate masks ==========
class PrefixCollator:
    """
        Align prefixes of different lengths to the same length:
      - Within a batch, pad to the maximum sequence length of that batch, or
        truncate to `max_len` (keeping the most recent tokens).
      - Padding positions have action = 0, but are masked out by `attn_mask`.
    """
    def __init__(self, max_len: Optional[int] = 256, pad_value: int = -1):
        self.max_len = max_len
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict]):
        B = len(batch)
        lens = [len(x["prefix_resources"]) for x in batch]
        T = max(lens) if self.max_len is None else min(max(lens), self.max_len)

        prefix_res = torch.full((B, T), self.pad_value, dtype=torch.long)
        prefix_act = torch.zeros((B, T), dtype=torch.long)  # pad位值无关紧要
        attn_mask  = torch.zeros((B, T), dtype=torch.bool)  # True=有效
        user_ids   = torch.zeros(B, dtype=torch.long)
        last_res   = torch.zeros(B, dtype=torch.long)
        labels     = torch.zeros(B, dtype=torch.float32)

        for i, s in enumerate(batch):
            r, a = s["prefix_resources"], s["prefix_actions"]
            if len(r) > T:
                r = r[-T:]
                a = a[-T:]
            L = len(r)
            prefix_res[i, :L] = torch.tensor(r, dtype=torch.long)
            prefix_act[i, :L] = torch.tensor(a, dtype=torch.long)
            attn_mask[i, :L] = True
            user_ids[i] = s["user_id"]
            last_res[i] = s["last_resource"]
            labels[i]   = float(s["label"])

        return {
            "user_ids": user_ids,                    # (B,)
            "prefix_resources": prefix_res,          # (B,T)
            "prefix_actions": prefix_act,            # (B,T)
            "attn_mask": attn_mask,                  # (B,T) True=valid token
            "last_resource": last_res,               # (B,)
            "labels": labels                         # (B,)
        }


# ========== User Sample → (Prefix, Last Step, Label) ==========
class LastStepSupervisionDataset(Dataset):
    """
    For each user sample:
      Input:  the first L−1 pairs of (resource_id, action)
      Label:  the L-th action; during classification, the L-th resource_id is also used.
    Samples with history length < 2 will be skipped.
    Each element in base[i] must have the following structure:
      {'user_id': int, 'resources': List[int], 'actions': List[int]}
    """
    def __init__(self, base: Dataset):
        self.base = base
        self.valid_indices = [i for i in range(len(base)) if len(base[i]["resources"]) >= 2]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        s = self.base[self.valid_indices[idx]]
        user_id = s["user_id"]
        resources = s["resources"]
        actions = s["actions"]
        L = len(resources)
        return {
            "user_id": user_id,
            "prefix_resources": resources[:L-1],
            "prefix_actions": actions[:L-1],
            "last_resource": resources[L-1],
            "label": actions[L-1]
        }


#%%
def batch_user_resource_embeddings(
    samples: List[Dict],
    user_embedding: np.ndarray,
    resource_embedding: np.ndarray,
    max_len: int = None,
    device: str = None,      # automatically detect GPU
):
    """
    Pack a batch of samples into:
      - user_vecs:   (B, d_gnn)
      - res_vecs:    (B, T, d_gnn)
      - labels:      (B, T)
      - attn_mask:   (B, T)
    """
    # Automatically select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    B = len(samples)
    lens = [len(s["resources"]) for s in samples]
    T = max(lens) if max_len is None else min(max(lens), max_len)
    d_gnn = user_embedding.shape[1]

    user_vecs = torch.zeros(B, d_gnn, device=device)
    res_vecs  = torch.zeros(B, T, d_gnn, device=device)
    labels    = torch.full((B, T), fill_value=-1, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i, s in enumerate(samples):
        uid = s["user_id"]
        res_ids = s["resources"]
        acts = s["actions"]
        if len(res_ids) > T:
            res_ids = res_ids[-T:]
            acts = acts[-T:]
        L = len(res_ids)

        user_vecs[i] = torch.from_numpy(user_embedding[uid]).to(device)
        res_vecs[i, :L] = torch.from_numpy(resource_embedding[np.array(res_ids)]).to(device)
        labels[i, :L] = torch.tensor(acts, dtype=torch.long, device=device)
        attn_mask[i, :L] = True

    return user_vecs, res_vecs, labels, attn_mask


#%%
def split_userhistorydataset_by_user(dataset,
                                     train_ratio=0.8,
                                     val_ratio=0.1,
                                     seed=42,
                                     out_dir="./data"):
    """
    Randomly split a UserHistoryDataset by user into three subsets: train/val/test,
    and automatically save them as .pt files.

    Saved file format:
        ./data/train_dataset.pt
        ./data/val_dataset.pt
        ./data/test_dataset.pt
    """
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(seed)
    user_ids = list(dataset.user_histories.keys())
    rng.shuffle(user_ids)

    n_total = len(user_ids)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_users = set(user_ids[:n_train])
    val_users   = set(user_ids[n_train:n_train + n_val])
    test_users  = set(user_ids[n_train + n_val:])

    all_indices = list(range(len(dataset)))
    train_idx = [i for i in all_indices if dataset.user_ids[i] in train_users]
    val_idx   = [i for i in all_indices if dataset.user_ids[i] in val_users]
    test_idx  = [i for i in all_indices if dataset.user_ids[i] in test_users]

    print(f"Total number of users: {n_total}")
    print(f"Train: {len(train_idx)} users, Val: {len(val_idx)} users, Test: {len(test_idx)} users")

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    # --- Save as .pt files ---
    def extract_subset_data(subset):
        return [subset.dataset[i] for i in subset.indices]

    train_path = os.path.join(out_dir, "train_dataset.pt")
    val_path   = os.path.join(out_dir, "val_dataset.pt")
    test_path  = os.path.join(out_dir, "test_dataset.pt")

    torch.save(extract_subset_data(train_set), train_path)
    torch.save(extract_subset_data(val_set), val_path)
    torch.save(extract_subset_data(test_set), test_path)

    print(f" Datasets successfully saved to：\n  {train_path}\n  {val_path}\n  {test_path}")

    return train_set, val_set, test_set



#%%
import torch
from tqdm import tqdm

def remap_and_filter_dataset(dataset,
                             mapping_path="./data/id_mappings.pt",
                             save_path="./data/dataset_remapped.pt"):
    """
    Remap user_id and resource_id in the dataset to new IDs according to the mapping table.
    If a user or resource does not exist in the mapping, that sample will be discarded.
    The cleaned dataset will be saved as a new .pt file.
    """
    mappings = torch.load(mapping_path)
    u_map = mappings["user_real2mapped"]
    r_map = mappings["resource_real2mapped"]

    cleaned_data = []
    missing_users = 0
    missing_resources = 0

    for i in tqdm(range(len(dataset)), desc="Filtering & Remapping"):
        sample = dataset[i]
        uid = sample["user_id"]
        res_ids = sample["resources"]
        acts = sample["actions"]

        # 跳过没有映射的用户
        if uid not in u_map:
            missing_users += 1
            continue

        mapped_resources = []
        skip = False
        for r in res_ids:
            if r not in r_map:
                missing_resources += 1
                skip = True
                break
            mapped_resources.append(r_map[r])

        if skip:
            continue

        cleaned_data.append({
            "user_id": u_map[uid],
            "resources": mapped_resources,
            "actions": acts
        })

    print(f"Successfully retained {len(cleaned_data)} samples.")
    print(f"Removed {missing_users} samples with missing users and {missing_resources} samples with missing resources.")

    # 保存为 .pt 文件
    torch.save(cleaned_data, save_path)
    print(f"Saved to:  {save_path}")

    # --- 转换为新的 UserHistoryDataset ---
    # UserHistoryDataset 需要一个 DataFrame，所以我们先重构一个
    rows = []
    for s in cleaned_data:
        u = s["user_id"]
        for r, a in zip(s["resources"], s["actions"]):
            rows.append({"PERSON_ID": u, "TARGET_NAME": r, "ACTION": a})
    df = pd.DataFrame(rows)

    new_dataset = UserHistoryDataset(df)
    print(f"New UserHistoryDataset created with {len(new_dataset)} users.")

    return new_dataset



#%%

def save_id_mappings_pt(acess_control_metadata_dict_remapping: dict,
                        out_path: str = "./id_mappings.pt",
                        user_key: str = "User_features",
                        resource_key: str = "Resource_features",
                        user_real_col: str = None,
                        user_mapped_col: str = None,
                        res_real_col: str = None,
                        res_mapped_col: str = None):
    """
    Extract user and resource ID mappings from `acess_control_metadata_dict_remapping`
    and save them into a single .pt file (PyTorch format).

    The saved file contains a dictionary with the following structure:
        {
          "user_real2mapped": dict,
          "user_mapped2real": dict,
          "resource_real2mapped": dict,
          "resource_mapped2real": dict
        }
    """

    def _extract_mapping(df: pd.DataFrame, real_col=None, mapped_col=None):
        if real_col is None:
            real_col = df.columns[0]
        if mapped_col is None:
            mapped_col = df.columns[-1]
        df = df[[real_col, mapped_col]].dropna().drop_duplicates()

        real2mapped = dict(zip(df[real_col].tolist(), df[mapped_col].tolist()))
        mapped2real = dict(zip(df[mapped_col].tolist(), df[real_col].tolist()))
        return real2mapped, mapped2real

    user_df = acess_control_metadata_dict_remapping[user_key]
    res_df = acess_control_metadata_dict_remapping[resource_key]

    user_real2mapped, user_mapped2real = _extract_mapping(
        user_df, user_real_col, user_mapped_col)
    resource_real2mapped, resource_mapped2real = _extract_mapping(
        res_df, res_real_col, res_mapped_col)

    all_mappings = {
        "user_real2mapped": user_real2mapped,
        "user_mapped2real": user_mapped2real,
        "resource_real2mapped": resource_real2mapped,
        "resource_mapped2real": resource_mapped2real,
    }

    # Save in PyTorch format
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(all_mappings, out_path)

    print(f"Mapping file saved to: {out_path}")
    print(f"Contains keys: {list(all_mappings.keys())}")
    return all_mappings


    #%% Build a dataset from history_df.
    # Specifically, group the data by user ID and, for each user, save all historical sequences
    # of resource IDs along with their corresponding labels (actions).
    def __init__(self, history_df: pd.DataFrame):
        """
        Build a dataset grouped by PERSON_ID.
        For each user, save all the history sequence of resource ids (TARGET_NAME)
        and the corresponding labels (ACTION).

        Args:
            history_df: DataFrame with columns ['ACTION', 'TARGET_NAME', 'PERSON_ID']
        """
        self.user_histories = {}
        grouped = history_df.groupby('PERSON_ID')
        for user_id, group in grouped:
            # Sort by row order (if time order not explicit)
            seq_resources = group['TARGET_NAME'].tolist()
            seq_actions = group['ACTION'].tolist()
            self.user_histories[user_id] = {
                'resources': seq_resources,
                'actions': seq_actions
            }

        # Save all user ids for indexing
        self.user_ids = list(self.user_histories.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        history = self.user_histories[user_id]
        return {
            'user_id': user_id,
            'resources': history['resources'],
            'actions': history['actions']
        }

