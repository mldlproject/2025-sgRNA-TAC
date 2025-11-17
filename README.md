# Predicting sgRNA Off-Target Effects Using DNA-Language Embedding and Triplet Attention Mechanism

### Nhat-Anh Nguyen-Dang, Phuc H. Le, Thanh-Hoang Nguyen-Vo, Binh P. Nguyen  

## Abstract

CRISPR-Cas9 has transformed genome editing by enabling precise DNA modifications, but off-target effects remain a significant challenge. This study presents a novel deep learning framework that integrates a DNABERT-6 pretrained sequence encoder with an attention-based convolutional neural network and a customized Triplet Attention mechanism to predict sgRNA off-target effects. Our model processes wild-type and mutated sequence pairs to learn intricate sequence relationships, with the Triplet Attention module effectively emphasizing critical regions. Comparative evaluations against 28 traditional machine learning models and several state-of-the-art deep learning methods demonstrate the superiority of the proposed approach, achieving an Area Under the Receiver Operating Characteristic curve of 0.8954 and an Area Under the Precision-Recall Curve of 0.8798 on the test set. Ablation studies confirm the impact of the Triplet Attention mechanism, while robustness assessments over 30 independent trials show consistently high performance. These results highlight the potential of combining DNA-language embeddings with advanced attention mechanisms to improve the accuracy and reliability of off-target prediction, contributing to safer and more effective CRISPR applications.

## Installation

```bash
git clone https://github.com/mldlproject/sgRNA-TAC.git
cd sgRNA-TAC
python -m venv .venv
.\.venv\Scripts\activate      # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

## Usage

### Training script (includes evaluation & file export)
```bash
python train.py --config configs/default.yaml
```

## Data

Download the PKD off-target dataset (wild-type and mutated sequences with Day21-ETP labels) and place it in `data/PKD.csv`. The current folder structure expects columns `WTSequence (WildType)`, `MutatedSequence`, and `Day21-ETP-binarized`. Additional FASTA exports reside under `data/Fasta_PKD/`. Replace with your own dataset if needed, keeping the column names consistent.

## Citation

```bibtex
@article{nguyen2025sgRNA,
  title={Predicting sgRNA Off-Target Effects Using DNA-Language Embedding and Triplet Attention Mechanism},
  author={Nhat-Anh Nguyen-Dang and Phuc H. Le and Thanh-Hoang Nguyen-Vo and Binh P. Nguyen},
  year={2025},
  note={Manuscript is Under-review}
}
```

## Contact
Free to reach out to me via [email](thanhhoang.nguyenvo@ou.edu.vn).