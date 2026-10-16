#!/usr/bin/env python
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from tqdm import tqdm
import random
import argparse
from model import MODEL
from utils import Seed_everything, Metric, Write_log, TaskDataset
from focalLoss import FocalLoss, DiceLoss, CombinedLoss

warnings.simplefilter("ignore")


def train_model(
    train_df,
    val_df,
    protein_data,
    model_class,
    config,
    fold,
    output_root="./output/",
    args=None,
):
    output_weight = output_root + "weight/"
    if not os.path.exists(output_weight):
        os.makedirs(output_weight)

    log = open(output_weight + f"fold{fold}.log", "w", buffering=1)
    Write_log(log, str(config) + "\n")

    node_features = config["node_features"]
    edge_features = config["edge_features"]
    hidden_dim = config["hidden_dim"]
    num_encoder_layers = config["num_encoder_layers"]
    k_neighbors = config["k_neighbors"]
    augment_eps = config["augment_eps"]
    dropout = config["dropout"]
    distance_alpha = config.get("distance_alpha", 0.5)  
    use_rbf_distance = config.get("use_rbf_distance", True)  
    num_rbf = config.get("num_rbf", 16) 
    use_cross_modal = config.get(
        "use_cross_modal", True
    )  
    use_dense_connection = config.get(
        "use_dense_connection", False
    )  
    use_gatv2 = config.get("use_gatv2", True)  
    use_gvp = config.get("use_gvp", False) 
    use_gated_fusion = config.get("use_gated_fusion", False)  
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = config["patience"]

    train_df_reset = train_df.reset_index(drop=True)
    val_df_reset = val_df.reset_index(drop=True)

    train_dataset = TaskDataset(train_df_reset, protein_data, "label")
    val_dataset = TaskDataset(val_df_reset, protein_data, "label")

    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=config["num_samples"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,  
        prefetch_factor=4,  
        pin_memory=True,  
        persistent_workers=True,  
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8, 
        prefetch_factor=4,  
        pin_memory=True, 
        persistent_workers=True, 
    )

    model = model_class(
        node_features,
        edge_features,
        hidden_dim,
        num_encoder_layers,
        k_neighbors,
        augment_eps,
        dropout,
        distance_alpha,
        use_gatv2=use_gatv2,
        use_rbf_distance=use_rbf_distance,
        num_rbf=num_rbf,
        use_cross_modal=use_cross_modal,
        use_dense_connection=use_dense_connection,
        use_gvp=use_gvp,
        use_gated_fusion=use_gated_fusion
    )
    model.cuda()

    # Use DataParallel if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    use_dice_loss = config.get("use_dice_loss", False)
    if use_dice_loss:
        criterion = CombinedLoss(
            focal_weight=config.get("focal_weight", 0.7),
            dice_weight=config.get("dice_weight", 0.3),
            focal_alpha=config.get("focal_alpha", 0.8),
            focal_gamma=config.get("focal_gamma", 2.0),
            dice_smooth=config.get("dice_smooth", 1.0),
        )
        Write_log(
            log,
            f"Using CombinedLoss: Focal({config.get('focal_weight', 0.7)}) + Dice({config.get('dice_weight', 0.3)})\n",
        )
    else:
        criterion = FocalLoss(
            alpha=config.get("focal_alpha", 0.8), gamma=config.get("focal_gamma", 2.0)
        )
        Write_log(log, "Using FocalLoss\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, 
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-2, 
    )


    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    best_epoch = 0
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for data in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training"
        ):
            _, protein_X, protein_node_features, protein_masks, y, adj = data

            protein_X = protein_X.cuda()
            protein_node_features = protein_node_features.cuda()
            protein_masks = protein_masks.cuda()
            y = y.cuda()
            adj = adj.cuda()

            optimizer.zero_grad()

            outputs = model(protein_X, protein_node_features, protein_masks, adj)

            loss = 0
            for i in range(len(protein_masks)):
                mask = protein_masks[i].bool()
                if mask.sum() > 0:  
                    loss += criterion(outputs[i][mask], y[i][mask])

            loss = loss / len(protein_masks)  

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                for i in range(len(protein_masks)):
                    mask = protein_masks[i].bool()
                    if mask.sum() > 0:
                        train_preds.append(probs[i][mask].cpu().numpy())
                        train_labels.append(y[i][mask].cpu().numpy())

        train_preds = np.concatenate([p.flatten() for p in train_preds])
        train_labels = np.concatenate([l.flatten() for l in train_labels])
        train_metrics = Metric(train_preds, train_labels)

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for data in tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"
            ):
                _, protein_X, protein_node_features, protein_masks, y, adj = data
    
                protein_X = protein_X.cuda()
                protein_node_features = protein_node_features.cuda()
                protein_masks = protein_masks.cuda()
                y = y.cuda()
                adj = adj.cuda()
  
                outputs = model(protein_X, protein_node_features, protein_masks, adj)

                loss = 0
                for i in range(len(protein_masks)):
                    mask = protein_masks[i].bool()
                    if mask.sum() > 0:
                        loss += criterion(outputs[i][mask], y[i][mask])

                loss = loss / len(protein_masks)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                for i in range(len(protein_masks)):
                    mask = protein_masks[i].bool()
                    if mask.sum() > 0:
                        val_preds.append(probs[i][mask].cpu().numpy())
                        val_labels.append(y[i][mask].cpu().numpy())

        val_preds = np.concatenate([p.flatten() for p in val_preds])
        val_labels = np.concatenate([l.flatten() for l in val_labels])
        val_metrics = Metric(val_preds, val_labels)

        log_message = (
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss / len(train_dataloader):.6f}, "
            f"Val Loss: {val_loss / len(val_dataloader):.6f}, "
            f"Train AUC: {train_metrics[0]:.6f}, "
            f"Val AUC: {val_metrics[0]:.6f}, "
            f"Train AUPRC: {train_metrics[1]:.6f}, "
            f"Val AUPRC: {val_metrics[1]:.6f}, "
            f"Train MCC: {train_metrics[2]:.6f}, "
            f"Val MCC: {val_metrics[2]:.6f}, "
            f"Train F1: {train_metrics[6]:.6f}, "
            f"Val F1: {val_metrics[6]:.6f}"
        )

        Write_log(log, log_message)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        Write_log(log, f"Learning rate: {current_lr:.6e}")

        current_val_auc = val_metrics[0] 
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch
            no_improvement = 0

            if isinstance(model, nn.DataParallel):
                torch.save(
                    model.module.state_dict(), output_weight + f"fold{fold}.ckpt"
                )
            else:
                torch.save(model.state_dict(), output_weight + f"fold{fold}.ckpt")

            Write_log(
                log,
                f"Model saved at epoch {epoch + 1} with validation AUC: {best_val_auc:.6f}",
            )
        else:
            no_improvement += 1

        if no_improvement >= patience:
            Write_log(
                log,
                f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch + 1} with validation AUC: {best_val_auc:.6f}",
            )
            break

    log.close()
    return best_val_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/")
    parser.add_argument("--feature_path", type=str, default="./feature_combined/")
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--task", type=str, default="PRO")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n_folds", type=int, default=5)

    args = parser.parse_args()

    args.node_features = 1570  
    args.distance_alpha = 0.5
    args.use_rbf_distance = True
    args.num_rbf = 16
    args.use_cross_modal = True
    args.use_dense_connection = True
    args.use_gatv2 = True
    args.use_gvp = True
    args.use_gated_fusion = False


    Seed_everything(seed=args.seed)


    train_df = pd.read_csv(args.dataset_path + "PRO_Train335.csv")

    protein_data = {}
    for pdb_id in tqdm(train_df["ID"].unique(), desc="Loading protein data"):
        try:
            protein_data[pdb_id] = (
                torch.load(args.feature_path + f"{pdb_id}_X.tensor"),
                torch.load(args.feature_path + f"{pdb_id}_node_feature.tensor"),
                torch.load(args.feature_path + f"{pdb_id}_mask.tensor"),
                torch.load(args.feature_path + f"{pdb_id}_label.tensor"),
                torch.load(args.feature_path + f"{pdb_id}_adj.tensor"),
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load data for {pdb_id}: {e}")
            continue


    if protein_data:
        sample_id = list(protein_data.keys())[0]
        detected_dim = protein_data[sample_id][1].shape[1]
        if detected_dim != args.node_features:
            raise ValueError(
                f"Node feature dimension mismatch: expected {args.node_features}, "
                f"got {detected_dim} from {args.feature_path}"
            )

    # Model configuration
    nn_config = {
        "node_features": args.node_features,
        "edge_features": 16,
        "hidden_dim": 128,
        "num_encoder_layers": 4,  
        "k_neighbors": 30,
        "augment_eps": 0.05,  
        "dropout": 0.18,  
        "distance_alpha": args.distance_alpha,  
        "use_rbf_distance": args.use_rbf_distance,  
        "num_rbf": args.num_rbf,  
        "use_cross_modal": args.use_cross_modal,
        "use_dense_connection": args.use_dense_connection,
        "use_gatv2": args.use_gatv2,
        "use_gvp": args.use_gvp,
        "use_gated_fusion": args.use_gated_fusion,
        "id_name": "ID",
        "obj_max": 1,
        "epochs": 40,
        "patience": 8, 
        "batch_size": 32,
        "num_samples": 335 * 5,
        "folds": args.n_folds,
        "seed": args.seed,
        "focal_alpha": 0.90,
        "focal_gamma": 2.0,
        "remark": args.task
        + " binding site prediction (DAGATv2="
        + str(args.use_gatv2)
        + ", GVP="
        + str(args.use_gvp)
        + ", RBF="
        + str(args.use_rbf_distance)
        + ")",
    }

    print("\n" + "=" * 80)
    print("Model Configuration")
    print("=" * 80)
    print(f"DAGATv2 branch: {'enabled' if args.use_gatv2 else 'disabled'}")
    print(f"GVP branch:     {'enabled' if args.use_gvp else 'disabled'}")
    print(f"Cross-Modal Attention: {'enabled' if args.use_cross_modal else 'disabled'}")
    print(f"Dense Connection: {'enabled' if args.use_dense_connection else 'disabled'}")
    print(f"RBF Distance: {'enabled' if args.use_rbf_distance else 'disabled'}")
    if args.use_gatv2 and args.use_gvp:
        print("Full model: DAGATv2 + GVP dual-branch")
    elif args.use_gatv2:
        print("Single-branch: DAGATv2 only")
    elif args.use_gvp:
        print("Single-branch: GVP only")
    else:
        print("Warning: all branches disabled")
    print("=" * 80 + "\n")

    output_root = args.output_path
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    if not os.path.exists(output_root + "weight/"):
        os.makedirs(output_root + "weight/")

    available_ids = list(protein_data.keys())
    train_df = train_df[train_df["ID"].isin(available_ids)]
    print(
        f"Using {len(available_ids)} proteins with complete feature data out of {len(train_df['ID'].unique())} total proteins"
    )

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(available_ids)):
        print(f"\n{'=' * 50}\nTraining Fold {fold + 1}/{args.n_folds}\n{'=' * 50}")

        train_ids = [available_ids[i] for i in train_idx]
        val_ids = [available_ids[i] for i in val_idx]

        fold_train_df = train_df[train_df["ID"].isin(train_ids)]
        fold_val_df = train_df[train_df["ID"].isin(val_ids)]

        fold_metric = train_model(
            fold_train_df,
            fold_val_df,
            protein_data,
            MODEL,
            nn_config,
            fold,
            output_root,
            args,
        )

        fold_metrics.append(fold_metric)

    # Print overall results
    print(f"\n{'=' * 50}\nCross-Validation Results\n{'=' * 50}")
    for fold, metric in enumerate(fold_metrics):
        print(f"Fold {fold + 1}: AUC = {metric:.6f}")
    print(f"Average AUC: {np.mean(fold_metrics):.6f} ± {np.std(fold_metrics):.6f}")


if __name__ == "__main__":
    main()
