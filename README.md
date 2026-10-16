# MFTGSite

MFTGSite is a dual-branch graph neural network for residue-level protein–protein interaction site prediction. It integrates ESM2, AlphaFold2 single representations (AF2-SR), DSSP, and BLOSUM62 features through cross-feature attention, and jointly models residue relationships using a DAGATv2 topological branch and a GVP geometric branch.

## System Requirements

```text
Python==3.8.20
torch==2.4.1
torchvision==0.19.1
transformers==4.46.3
biopython==1.79
scikit-learn==1.3.0
numpy==1.23.5
pandas==2.0.3
tqdm==4.67.1
joblib==1.4.2
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Feature Generation

### Sequence and structural feature generation

1. **ESM2** features are generated using [ESM2 (`esm2_t33_650M_UR50D`)](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
2. **AlphaFold2 single representations (AF2-SR)** are generated using the [ColabFold AlphaFold2 notebook](https://colab.research.google.com/github/sokrypton/ColabFold/blob/v1.5.5/batch/AlphaFold2_batch.ipynb) (ColabFold v1.5.5) with the AlphaFold2-ptm model, yielding 256-dimensional residue-level structural representations.
3. **DSSP** features are extracted from protein structures using [DSSP](https://github.com/PDB-REDO/dssp). Before using the software, grant execution permission with:

   ```bash
   chmod +x ./feature_generation/Software/dssp-2.0.4/mkdssp
   ```

4. **BLOSUM62** features are generated automatically during feature generation.

### Generate Features

After preparing the required protein sequences and structures, run:

```bash
cd feature_generation
python generate_all_features.py
cd ..
```

## Run MFTGSite for prediction

Pretrained five-fold model weights are provided in output/weight/. Three example proteins from Pro_Test_60 and their corresponding features are provided in infer_data/features/. To run prediction on these examples, use:

```bash
python test.py
```
