![Logo](images/cover.png)

## Next generation in lossless data compression

---
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-GNU--GPLv3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Fileformer** is a next-generation lossless neural network archiver that uses advanced transformer architecture and GPU-accelerated arithmetic compression. It compresses data more than twice as efficiently while running 10 times faster than cmix and nncp.


## Key features:

- LoRA-based adaptation – the neural network doesn't retrain for each file, but adapts on the fly using low-rank adaptation. This enables personalization without retraining the entire model.


- Long context – 32k bytes – the transformer sees 32k bytes of input data, allowing it to capture long-term dependencies.


- Hybrid architecture – Flash Attention + Multi-head Latent Attention (MLA) for high speed and efficient processing of long sequences on the GPU.


## Run Locally

1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or pip: pip install uv
```

2. Clone the repository

```bash
git clone https://github.com/Qwest1204/FileFormer
cd FileFormer
```

3. Download pre-trained weights from releases
```bash
wget -P checkpoint https://github.com/Qwest1204/FileFormer/releases/download/pre-alpha/model_enwiki_pre-v0.0.1.pt
```

4. Install project dependencies

```bash
uv sync
```

## Feedback

If you have any feedback, please reach out to us at workemailfordaniil@gmail.com

