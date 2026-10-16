import gc
import os
import re

import esm
import numpy as np
import torch
from tqdm import tqdm

_ESM2_MODEL_PATH = "/mnt/models/esm/esm2_t33_650M_UR50D.pt"


def getESM2(fasta_file: str, output_dir: str, gpu: str = "0") -> None:
    os.makedirs(output_dir, exist_ok=True)

    id_list, seq_list = [], []
    with open(fasta_file) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                id_list.append(line[1:])
            elif line and not all(c in "01" for c in line):
                seq_list.append(line)

    if not os.path.exists(_ESM2_MODEL_PATH):
        raise FileNotFoundError(
            f"ESM-2 model checkpoint not found: {_ESM2_MODEL_PATH}"
        )

    model, alphabet = esm.pretrained.load_model_and_alphabet(_ESM2_MODEL_PATH)
    last_layer = getattr(model, "num_layers", 33)
    gc.collect()

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() and gpu else "cpu"
    )
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"ESM-2 loaded on {device}. Processing {len(id_list)} sequences...")

    for pdb_id, sequence in tqdm(zip(id_list, seq_list), total=len(id_list)):
        out_path = os.path.join(output_dir, f"{pdb_id}.npy")
        if os.path.exists(out_path):
            continue

        sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        _, _, tokens = batch_converter([(pdb_id, sequence)])
        tokens = tokens.to(device)

        with torch.no_grad():
            result = model(tokens, repr_layers=[last_layer], return_contacts=False)

        L = len(sequence)
        emb = result["representations"][last_layer][0, 1 : L + 1].cpu().numpy()
        np.save(out_path, emb)
