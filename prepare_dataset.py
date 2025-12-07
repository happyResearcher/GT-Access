import pickle
import pandas as pd
from utility import (UserHistoryDataset,save_id_mappings_pt,remap_and_filter_dataset,
                     split_userhistorydataset_by_user
                     )

#%% load entire user-rsource interaction log history
history_df = pd.read_pickle('./data/access_log_original.pkl')
dataset = UserHistoryDataset(history_df)

#%% load user, resource static feature, and the mapping between real ID and node ID in GNN embedding
with open('./data/acess_control_metadata_dict_remapping.pickle', 'rb') as f:
    acess_control_metadata_dict_remapping = pickle.load(f)

mappings = save_id_mappings_pt(
    acess_control_metadata_dict_remapping,
    out_path="./data/id_mappings.pt",
    user_key="User_features",
    resource_key="Resource_features"
)

# map original id to gnn embedding id
new_dataset = remap_and_filter_dataset(
    dataset,
    mapping_path="./data/id_mappings.pt",
    save_path="./data/user_history_mappedID.pt"
)

# split dataset by user
train_set, val_set, test_set = split_userhistorydataset_by_user(
    new_dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    out_dir="./data"
)
