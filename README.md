# Multi-Modal-Medical-Image-Attention

Implementation of our published multi-modal attention framework for **medical image segmentation**, built on Transformer architectures from the Hugging Face ecosystem. The proposed method enables effective feature fusion across multiple imaging modalities while dynamically adapting to missing modalities by excluding them during computation.

📄 **Paper**: https://ieeexplore.ieee.org/document/11230342  
⚠️ **Note**: The dataset is not included due to privacy restrictions.

---

## Overview

Medical image segmentation plays a critical role in clinical diagnosis and treatment planning. Leveraging multiple imaging modalities (e.g., CT, MRI) provides complementary information that can significantly improve segmentation performance. However, many existing methods assume fixed modality availability or rely on imputation when modalities are missing.

This work introduces a **Transformer-based multi-modal attention framework** that performs adaptive cross-modal feature fusion. The model dynamically adjusts to the available modalities by **excluding missing inputs from computation**, allowing it to operate under varying modality conditions without introducing artificial signals such as zero-filling.

---

## Model Architecture

![Model Overview](docs/proposal.png)

The framework consists of:
- Modality-specific feature encoders  
- Transformer-based attention for cross-modal interaction  
- Dynamic exclusion of missing modalities during attention computation  
- A segmentation head for final prediction  

---

## Key Features

- Transformer-based multi-modal attention (Hugging Face backbone)  
- Designed for **medical image segmentation**  
- Native handling of missing modalities (no imputation required)  
- Dynamic multi-modal input dimension during training and inference  
- Robust cross-modal feature fusion  

---

## Use Case

This framework targets **multi-modal medical image segmentation**, where:

- Multiple modalities provide complementary information  
- Some modalities may be unavailable in real-world scenarios  

The model:
- Utilizes all available modalities when present  
- Maintains stable performance when modalities are missing  

---

## Installation

This project requires **GPU acceleration** with CUDA-enabled PyTorch.

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset used in our paper is **not publicly available** due to privacy restrictions.

To use this code:

- Prepare your own segmentation dataset  
- Provide dataset split files in CSV format:
  - `train.csv`
  - `val.csv`
  - `test.csv`

---

## CSV Format

Each dataset split must follow the structure below.

**Important:**
- Segmentation maps must be included in the same CSV  
- They are identified by appending **`L`** to the modality name  
  - Example: `am → amL`  

| Column        | Description                                      |
|--------------|--------------------------------------------------|
| Unnamed: 0   | Row index (optional, can be ignored)             |
| ID           | Patient or case identifier                       |
| Modality     | Imaging modality name                            |
| Data         | Image file name                                  |
| Data path    | Full or relative path to the image file          |

### Example

```csv
Unnamed: 0,ID,Modality,Data,Data path
0,15630104,am,1.png,dataset/resized_dataset_divisible/15630104/am/1.png
1,15630104,am,2.png,dataset/resized_dataset_divisible/15630104/am/2.png
2,15630104,amL,1.png,dataset/resized_map_divisible/15630104/amL/1.png
3,15630104,amL,2.png,dataset/resized_map_divisible/15630104/amL/2.png
```

---

## ⚠️ Modality Configuration (Important)

When using the dataset loader, you must explicitly define the list of **all available modalities** in your dataset.

Example:

```python
all_modalities = ['dc', 'ec', 'pc', 'am', 'tm']
```

---

## Usage

### 🔹 Training

```bash
python scripts/train.py \
  --modality tm \
  --csv_train <path_to_train_csv> \
  --csv_val <path_to_val_csv> \
  --save_dir ./outputs
```

### 🔹 Evaluation

```bash
  python scripts/evaluate.py \
  --modality tm \
  --csv <path_to_val_csv> \
  --checkpoint <path_to_model_checkpoint>
```

### 🔹 Inference

```bash
python scripts/inference.py \
  --modality tm \
  --csv <path_to_test_csv> \
  --checkpoint <path_to_model_checkpoint> \
  --output_dir ./outputs/preds
```