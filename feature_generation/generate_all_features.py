#!/usr/bin/env python

import argparse
import os
import pickle
import re

import joblib
import numpy as np
import torch
from tqdm import tqdm

from extract_adjacency import prepare_adj
from extract_blosum62 import get_blosum62_features_normalized
from extract_coordinates import PDBFeature
from extract_dssp import get_dssp
from extract_esm2 import getESM2
from feature_utils import parse_fasta_file


def create_directories():
    dirs = [
        "../ESM2raw",
        "../ESM2norm",
        "../SINGLE256norm",
        "../dssp",
        "../SC_position",
        "../feature_combined",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created or confirmed directory: {d}")


def detect_single_repr_dir(fasta_file):
    basename = os.path.basename(fasta_file)
    match = re.search(r'(\d+)', basename)

    if match:
        num = match.group(1)
        single_repr_dir = f"../datasets/AF2-SR/single_repr_{num}/"
        if os.path.exists(single_repr_dir):
            return single_repr_dir

    possible_dirs = [
        "../datasets/AF2-SR/single_repr_335/",
        "../datasets/AF2-SR/single_repr_60/",
        "../datasets/AF2-SR/single_repr_315/",
        "../datasets/AF2-SR/single_repr_31/",
    ]

    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"Warning: could not auto-detect single_repr dir, using {dir_path}")
            return dir_path

    raise FileNotFoundError(
        f"No matching single_repr directory found for {fasta_file}.\n"
        "Ensure ../datasets/AF2-SR/single_repr_XXX/ exists."
    )


def compute_normalization_params(data_dict, feature_dir, feature_name):
    print(f"\nComputing {feature_name} normalisation params from {len(data_dict)} proteins...")

    Max_list, Min_list = [], []

    for i, prot_id in enumerate(tqdm(data_dict.keys())):
        feature_file = os.path.join(feature_dir, f"{prot_id}.npy")

        if not os.path.exists(feature_file):
            print(f"Warning: {feature_file} not found, skipping")
            continue

        raw = np.load(feature_file)
        Max_list.append(np.max(raw, axis=0))
        Min_list.append(np.min(raw, axis=0))

        if i % 100 == 0 and i > 0:
            Max_list = [np.max(np.array(Max_list), axis=0)]
            Min_list = [np.min(np.array(Min_list), axis=0)]

    Max_param = np.max(np.array(Max_list), axis=0)
    Min_param = np.min(np.array(Min_list), axis=0)

    print(f"Computed {feature_name} normalisation params")
    print(f"  Min shape: {Min_param.shape}, range: [{Min_param.min():.2f}, {Min_param.max():.2f}]")
    print(f"  Max shape: {Max_param.shape}, range: [{Max_param.min():.2f}, {Max_param.max():.2f}]")

    return Min_param, Max_param


def normalize_features(data_dict, feature_dir_in, feature_dir_out, Min_param, Max_param, feature_name):
    print(f"\nNormalising {feature_name} features ({len(data_dict)} proteins)...")

    for prot_id in tqdm(data_dict.keys()):
        in_path = os.path.join(feature_dir_in, f"{prot_id}.npy")
        out_path = os.path.join(feature_dir_out, f"{prot_id}.npy")

        if not os.path.exists(in_path):
            print(f"Warning: {in_path} not found, skipping")
            continue

        raw = np.load(in_path)
        normalised = (raw - Min_param) / (Max_param - Min_param + 1e-8)
        np.save(out_path, normalised)

    print(f"Normalised {feature_name} features")


def save_normalization_params(params_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(params_dict, f)
    print(f"Saved normalisation params to {output_file}")


def load_normalization_params(params_file):
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
    print(f"Loaded normalisation params from {params_file}")
    return params


def normalize_single_repr(data_dict, single_repr_dir_train, single_repr_dir_test,
                          Min_param=None, Max_param=None, is_train=True):
    print(f"\nProcessing single_repr features ({'train' if is_train else 'test'})...")

    single_repr_dir = single_repr_dir_train if is_train else single_repr_dir_test

    if is_train:
        print("Computing single_repr normalisation params...")
        Max_list, Min_list = [], []
        missing_count = 0

        for i, prot_id in enumerate(tqdm(data_dict.keys())):
            # Try both filename conventions
            single_repr_file = os.path.join(single_repr_dir, f"{prot_id}_single_repr.npy")
            if not os.path.exists(single_repr_file):
                single_repr_file = os.path.join(single_repr_dir, f"{prot_id}.pkl")

            if not os.path.exists(single_repr_file):
                missing_count += 1
                continue

            try:
                if single_repr_file.endswith('.npy'):
                    single_repr = np.load(single_repr_file)
                else:
                    with open(single_repr_file, 'rb') as f:
                        single_repr = joblib.load(f)

                Max_list.append(np.max(single_repr, axis=0))
                Min_list.append(np.min(single_repr, axis=0))

                if i % 100 == 0 and i > 0:
                    Max_list = [np.max(np.array(Max_list), axis=0)]
                    Min_list = [np.min(np.array(Min_list), axis=0)]
            except Exception as e:
                print(f"Warning: failed to load {single_repr_file}: {e}")
                missing_count += 1
                continue

        if len(Max_list) == 0:
            raise ValueError(
                f"No valid single_repr files found in {single_repr_dir}\n"
                f"Expected: {len(data_dict)}, missing: {missing_count}"
            )

        Max_param = np.max(np.array(Max_list), axis=0)
        Min_param = np.min(np.array(Min_list), axis=0)

        print(f"Computed single_repr normalisation params")
        print(f"  Loaded: {len(data_dict) - missing_count}/{len(data_dict)} files")
        if missing_count > 0:
            print(f"  Missing: {missing_count} files")
        print(f"  Min shape: {Min_param.shape}, range: [{Min_param.min():.2f}, {Min_param.max():.2f}]")
        print(f"  Max shape: {Max_param.shape}, range: [{Max_param.min():.2f}, {Max_param.max():.2f}]")

    # Normalise and save to SINGLE256norm
    print(f"Normalising single_repr features...")
    for prot_id in tqdm(data_dict.keys()):
        single_repr_file = os.path.join(single_repr_dir, f"{prot_id}_single_repr.npy")
        if not os.path.exists(single_repr_file):
            single_repr_file = os.path.join(single_repr_dir, f"{prot_id}.pkl")

        if not os.path.exists(single_repr_file):
            continue

        if single_repr_file.endswith('.npy'):
            single_repr = np.load(single_repr_file)
        else:
            with open(single_repr_file, 'rb') as f:
                single_repr = joblib.load(f)

        single_repr_norm = (single_repr - Min_param) / (Max_param - Min_param + 1e-8)
        np.save(f"../SINGLE256norm/{prot_id}.npy", single_repr_norm)

    print(f"Normalised single_repr features")

    if is_train:
        return Min_param, Max_param
    else:
        return None, None


def prepare_features_combined(pdb_id, label, max_len, sequence):
    with open(f"../SC_position/{pdb_id}_psepos_SC.pkl", "rb") as f:
        X = joblib.load(f)

    actual_seq_len = X.shape[0]
    target_len = max_len  # enforce uniform length

    # Load normalised features
    esm2 = np.load(f"../ESM2norm/{pdb_id}.npy")
    single_repr = np.load(f"../SINGLE256norm/{pdb_id}.npy")
    dssp = np.load(f"../dssp/{pdb_id}.npy")
    blosum62 = get_blosum62_features_normalized(sequence)

    # Truncate or pad all features to target_len
    def _pad_or_truncate(arr, target):
        if arr.shape[0] > target:
            return arr[:target]
        elif arr.shape[0] < target:
            pad_len = target - arr.shape[0]
            return np.pad(arr, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        return arr

    esm2 = _pad_or_truncate(esm2, target_len)
    single_repr = _pad_or_truncate(single_repr, target_len)
    dssp = _pad_or_truncate(dssp, target_len)
    blosum62 = _pad_or_truncate(blosum62, target_len)
    X = _pad_or_truncate(X, target_len)

    # Verify uniform length
    seq_len = target_len
    assert esm2.shape[0] == seq_len, f"ESM-2 length mismatch: {esm2.shape[0]} vs {seq_len}"
    assert single_repr.shape[0] == seq_len, f"single_repr length mismatch: {single_repr.shape[0]} vs {seq_len}"
    assert dssp.shape[0] == seq_len, f"DSSP length mismatch: {dssp.shape[0]} vs {seq_len}"
    assert blosum62.shape[0] == seq_len, f"BLOSUM62 length mismatch: {blosum62.shape[0]} vs {seq_len}"
    assert X.shape[0] == seq_len, f"Coordinate length mismatch: {X.shape[0]} vs {seq_len}"

    # Concatenate node features: [L, 1280+256+14+20] = [L, 1570]
    node_feature = np.concatenate([esm2, single_repr, dssp, blosum62], axis=1)

    # No additional padding needed (seq_len == max_len by construction)
    node_feature_padded = node_feature
    X_padded = X

    # Create mask: 1 for actual residues, 0 for padding
    mask = np.zeros(max_len, dtype=np.float32)
    effective_len = min(actual_seq_len, max_len)
    mask[:effective_len] = 1.0

    # Process labels: truncate or pad to max_len
    label_array = np.array([int(c) for c in label], dtype=np.float32)
    if label_array.shape[0] > max_len:
        label_padded = label_array[:max_len]
    elif label_array.shape[0] < max_len:
        pad_len_label = max_len - label_array.shape[0]
        label_padded = np.pad(label_array, (0, pad_len_label), mode='constant', constant_values=0)
    else:
        label_padded = label_array

    # Save all tensors
    output_dir = "../feature_combined/"
    torch.save(torch.tensor(node_feature_padded, dtype=torch.float),
               f"{output_dir}{pdb_id}_node_feature.tensor")
    torch.save(torch.tensor(X_padded, dtype=torch.float),
               f"{output_dir}{pdb_id}_X.tensor")
    torch.save(torch.tensor(mask, dtype=torch.float),
               f"{output_dir}{pdb_id}_mask.tensor")
    torch.save(torch.tensor(label_padded, dtype=torch.float),
               f"{output_dir}{pdb_id}_label.tensor")


def process_dataset(fasta_file, is_train=True, norm_params=None, gpu="0"):
    print("\n" + "=" * 80)
    print(f"Processing dataset: {fasta_file}")
    print(f"Type: {'train' if is_train else 'test'}")
    print("=" * 80)

    data = parse_fasta_file(fasta_file)
    print(f"Parsed {len(data)} proteins from FASTA")

    # Step 1: Extract ESM-2 features
    print("\n[Step 1/7] Extracting ESM-2 features...")
    getESM2(fasta_file, "../ESM2raw/", gpu)
    print("ESM-2 extraction complete")

    # Step 2: Normalise ESM-2
    print("\n[Step 2/7] Normalising ESM-2 features...")
    if is_train:
        esm2_min, esm2_max = compute_normalization_params(data, "../ESM2raw/", "ESM-2")
        normalize_features(data, "../ESM2raw/", "../ESM2norm/", esm2_min, esm2_max, "ESM-2")
        esm2_params = {'min': esm2_min, 'max': esm2_max}
    else:
        if norm_params is None or 'esm2' not in norm_params:
            raise ValueError("Test set requires training-set normalisation params")
        esm2_params = norm_params['esm2']
        normalize_features(data, "../ESM2raw/", "../ESM2norm/",
                           esm2_params['min'], esm2_params['max'], "ESM-2")

    # Step 3: Normalise single_repr
    print("\n[Step 3/7] Normalising single_repr features...")

    single_repr_dir = detect_single_repr_dir(fasta_file)
    print(f"Detected single_repr directory: {single_repr_dir}")

    if is_train:
        single_min, single_max = normalize_single_repr(
            data, single_repr_dir, single_repr_dir, is_train=True
        )
        single_params = {'min': single_min, 'max': single_max}
    else:
        if norm_params is None or 'single_repr' not in norm_params:
            raise ValueError("Test set requires training-set normalisation params")
        single_params = norm_params['single_repr']
        normalize_single_repr(
            data, "../datasets/AF2-SR/single_repr_335/", single_repr_dir,
            Min_param=single_params['min'],
            Max_param=single_params['max'],
            is_train=False
        )

    # Step 4: Extract DSSP features
    print("\n[Step 4/7] Extracting DSSP features...")
    for prot_id in tqdm(data.keys()):
        get_dssp(prot_id, data[prot_id][0])
    print("DSSP extraction complete")

    # Step 5: Extract 3D coordinates
    print("\n[Step 5/7] Extracting 3D coordinates...")
    for prot_id in tqdm(data.keys()):
        PDBFeature(prot_id, "../datasets/pro_pro_pdb", "../SC_position")
    print("3D coordinate extraction complete")

    # Step 6: Generate adjacency matrices
    print("\n[Step 6/7] Generating adjacency matrices...")
    for prot_id in tqdm(data.keys()):
        prepare_adj(prot_id, 869)
    print("Adjacency matrix generation complete")

    # Step 7: Combine all features
    print("\n[Step 7/7] Combining all features...")
    skipped_count = 0
    for prot_id in tqdm(data.keys()):
        required_files = [
            f"../ESM2norm/{prot_id}.npy",
            f"../SINGLE256norm/{prot_id}.npy",
            f"../dssp/{prot_id}.npy",
            f"../SC_position/{prot_id}_psepos_SC.pkl"
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            skipped_count += 1
            if skipped_count <= 5:
                print(f"\nSkipping {prot_id}: missing files")
                for f in missing_files:
                    print(f"  - {f}")
            continue

        prepare_features_combined(prot_id, data[prot_id][1], 869, data[prot_id][0])

    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} proteins due to missing files")
    print("Feature combination complete")

    if is_train:
        return {
            'esm2': esm2_params,
            'single_repr': single_params
        }
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate all protein features for MFTGSite (data-leakage-free).'
    )
    parser.add_argument('--train_fasta', type=str,
                       default='../datasets/PRO_Train_335.fa',
                       help='Path to training-set FASTA file')
    parser.add_argument('--test_fasta', type=str,
                       action='append',
                       help='Path to test-set FASTA file (can be specified multiple times)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='CUDA device index')
    parser.add_argument('--norm_params_file', type=str,
                       default='../normalization_params.pkl',
                       help='Path to normalisation parameter file')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training-set processing (only process test sets)')

    args = parser.parse_args()

    if args.test_fasta is None:
        args.test_fasta = ['../datasets/PRO_Test_60.fa']

    print(f"\nGenerating features: {args.train_fasta} + {len(args.test_fasta)} test sets")
    create_directories()

    # Process training set
    if not args.skip_train:
        norm_params = process_dataset(
            args.train_fasta,
            is_train=True,
            gpu=args.gpu
        )
        save_normalization_params(norm_params, args.norm_params_file)
    else:
        if not os.path.exists(args.norm_params_file):
            raise FileNotFoundError(
                f"Normalisation parameter file not found: {args.norm_params_file}\n"
                "Process the training set first or provide the correct path."
            )
        norm_params = load_normalization_params(args.norm_params_file)

    # Process all test sets
    for i, test_fasta in enumerate(args.test_fasta, 1):
        print(f"\nProcessing test set {i}/{len(args.test_fasta)}: {test_fasta}")
        process_dataset(
            test_fasta,
            is_train=False,
            norm_params=norm_params,
            gpu=args.gpu
        )

    print("\n" + "=" * 60)
    print("Feature generation complete")
    print("=" * 60)

    if not args.skip_train:
        train_data = parse_fasta_file(args.train_fasta)
        print(f"Training: {len(train_data)} proteins")

    total_test = sum(len(parse_fasta_file(f)) for f in args.test_fasta if os.path.exists(f))
    print(f"Test: {total_test} proteins across {len(args.test_fasta)} sets")
    print(f"Output: feature_combined/")
    print("=" * 60)


if __name__ == '__main__':
    main()

