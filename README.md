# Multi-Modal-Medical-Image-Attention

Implementation of our published multi-modal attention framework for **medical image segmentation**, built on Transformer architectures from the Hugging Face ecosystem. The proposed method enables effective feature fusion across multiple imaging modalities while dynamically adapting to missing modalities by excluding them during computation.

📄 **Paper**: https://ieeexplore.ieee.org/document/11230342  
⚠️ **Note**: The dataset is not included due to privacy restrictions.

---

## Overview
Medical image segmentation is a fundamental task for clinical diagnosis and treatment planning. Leveraging multiple imaging modalities (e.g., CT, MRI) can significantly improve segmentation performance by providing complementary information. However, many existing approaches assume fixed modality availability or rely on imputation when modalities are missing.

In this work, we propose a **Transformer-based multi-modal attention framework** that performs adaptive cross-modal feature fusion. The model dynamically adjusts to the available modalities by **excluding missing inputs from computation**, allowing it to operate under varying modality conditions without introducing artificial signals such as zero-filling.

---

## Model Architecture

![Model Overview](docs/proposal.png)

The framework consists of:
- Modality-specific feature encoding
- Transformer-based attention for cross-modal interaction
- Dynamic exclusion of missing modalities during attention computation
- A segmentation head for final prediction

---

## Key Features
- Transformer-based multi-modal attention (Hugging Face backbone)
- Designed primarily for **medical image segmentation**
- Native handling of missing modalities (no imputation or masking tricks)
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

## Dataset
The dataset used in our paper is **not publicly available** due to privacy restrictions.

To use this code:
- Prepare your own segmentation dataset
- Provide dataset split files in CSV format:
  - `train.csv`
  - `val.csv`
  - `test.csv`

Each CSV file should include:
- Sample identifiers
- Paths to modality images

Ensure that:
- Modalities are properly aligned per sample
- Missing modalities are consistently represented (e.g., empty or null entries)

---

## Usage
```bash
python train.py \
  --csv_train <path_to_train_csv> \
  --csv_val <path_to_val_csv> \
  --csv_test <path_to_test_csv>