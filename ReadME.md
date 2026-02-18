# Building the Transformer from Scratch based on the famous "Attention is All You Need" paper

A PyTorch implementation of the Transformer architecture from the seminal 2017 paper. This project reproduces the base and big models for machine translation tasks. Reading the paper multiple times and asking their kids ("Claude and ChatGPT") doubts to understand how this archuitecture was implemented made me start this project. This is my attempt on implemnting this transformer.

## Installation

### Initial setup
```bash
git clone https://github.com/yourusername/transformer-implementation
cd transformer-implementation
pip install -r requirements.txt
```

### Training

```bash
python src/train.py --config configs/base_config.yaml
```
### Inference

```bash
python src/translate.py --sentence "Your sentence here"
```