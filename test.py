import os
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch

from model import MODEL
from utils import Metric, Seed_everything

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_DATA_DIR = os.path.join(ROOT_DIR, "infer_data")
WEIGHT_DIR = os.path.join(ROOT_DIR, "output", "weight") + os.sep  
RESULT_DIR = os.path.join(INFER_DATA_DIR, "prediction") + os.sep  
EXAMPLE_IDS = ["1qa9A", "3uvjA", "1wpxA"]
N_FOLDS = 5

SEED = 2024
MODEL_KWARGS = dict(
    node_features=1570,
    edge_features=16,
    hidden_dim=128,
    num_encoder_layers=4,
    k_neighbors=30,
    augment_eps=0.05,
    dropout=0.18,
    distance_alpha=0.5,
    use_gatv2=True,
    use_rbf_distance=True,
    num_rbf=16,
    use_cross_modal=True,
    use_dense_connection=True,
    use_gvp=True,
    use_gated_fusion=False,
)


def load_models():
    models = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(WEIGHT_DIR, f"fold{fold}.ckpt")
        model = MODEL(**MODEL_KWARGS)
        model.cuda()
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
        model.eval()
        models.append(model)
    return models


def main():
    Seed_everything(seed=SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)

    full_test_df = pd.read_csv(os.path.join(ROOT_DIR, "datasets", "PRO_Test60.csv"))
    test_df = full_test_df[full_test_df["ID"].isin(EXAMPLE_IDS)].reset_index(drop=True)

    print("=" * 80)
    print("MFTGSite inference example")
    print("=" * 80)
    print(f"Proteins: {', '.join(test_df['ID'])}")
    print(f"Weight path: {WEIGHT_DIR}")
    print("=" * 80)

    models = load_models()
    print(f"Loaded {len(models)} fold models")

    feature_path = os.path.join(INFER_DATA_DIR, "features") + os.sep
    all_preds, all_labels, rows = [], [], []

    with torch.no_grad():
        for pdb_id in test_df["ID"]:
            X = torch.load(feature_path + f"{pdb_id}_X.tensor").unsqueeze(0).cuda()
            node_feat = torch.load(feature_path + f"{pdb_id}_node_feature.tensor").unsqueeze(0).cuda()
            mask = torch.load(feature_path + f"{pdb_id}_mask.tensor").unsqueeze(0).cuda()
            label = torch.load(feature_path + f"{pdb_id}_label.tensor").unsqueeze(0).cuda()
            adj = torch.load(feature_path + f"{pdb_id}_adj.tensor").unsqueeze(0).cuda()

            outputs = torch.stack(
                [model(X, node_feat, mask, adj).sigmoid() for model in models], dim=0
            ).mean(0)

            seq_len = int(mask.sum().item())
            preds = outputs[0, :seq_len].cpu().numpy()
            labels = label[0, :seq_len].cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            rows.append({"ID": pdb_id, "prediction": preds.tolist()})

    preds_flat = np.concatenate(all_preds)
    labels_flat = np.concatenate(all_labels)
    auc, auprc, mcc, acc, pre, rec, f1 = Metric(preds_flat, labels_flat)

    print(
        "test_auc:%.6f, test_auprc:%.6f, test_mcc:%.6f, test_acc:%.6f, "
        "test_pre:%.6f, test_rec:%.6f, test_f1:%.6f"
        % (auc, auprc, mcc, acc, pre, rec, f1)
    )

    pd.DataFrame(rows).to_csv(RESULT_DIR + "result.csv", index=False)
    print(f"\nDone. Predictions saved to {RESULT_DIR}result.csv")


if __name__ == "__main__":
    main()
