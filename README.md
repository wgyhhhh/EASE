<h1 align="center">
<em>EASE</em>: Towards Real-Time Fake News Detection under Evidence Scarcity
</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.11277-b31b1b.svg)](https://arxiv.org/abs/2510.11277)
<img align="right" alt="ReaL" src="/assets/fake.png" width="40%">

EASE is an open-source, knowledge-augmented verification system for real-time fake news. EASE introduces a sequential evaluation mechanism comprising three independent perspectives: (1) Evidence-based evaluation, which assesses evidence and incorporates it into decision-making only when the evidence is sufficiently supportive; (2) Reasoning-based evaluation, which leverages the world knowledge of large language models (LLMs) and applies them only when their reliability is adequately established; and (3) Sentiment-based fallback, which integrates sentiment cues when neither evidence nor reasoning is reliable. 

We publicly release all implementation details, including training code, datasets, and infrastructure, to enable result verification and contribute to the research community.

**EASE Highlights**

- ⏱️ Real-time: EASE leverages a multi-perspective approach to fake news detection by integrating real-time evidence retrieval from the web, logical reasoning for factuality assessment, and sentiment analysis of textual characteristics.

- 🛡️ Robustness: For each retrieved piece of evidence, EASE assesses its reliability across multiple dimensions, applying the same rigor to reasoning and knowledge to assign the most appropriate supplementary information to each news item.

- 📄 Explainability: In addition to verifying the authenticity of news, EASE enhances transparency and interpretability by offering detailed explanations and the reasoning behind them.

## 📰 News

<details>
<summary><b>📋 Previous Releases</b></summary>

</details>

## 👀 About RealTimeNews-2025

Conventional fake news datasets often comprise news that is several years old. Such instances are grounded in a wealth of post-hoc evidence, including public discussions, official statements, and scientific articles. To advance research on real-time fake news detection, we introduce a new benchmark, RealTimeNews-25, consisting of 3,487 news articles collected between June 2024 and September 2025. The dataset covers recent and rapidly evolving events characterized by limited supporting evidence, providing a challenging and timely benchmark for evaluating model robustness in real-world, time-sensitive scenarios.

### ⬇ Download

We are currently organizing RealTimeNews-2025, which will be released via links on Baidu Netdisk and Google Drive.

### ✨ Dataset Examples

<img src="assets/Realtimenews.png" class="floatpic">

### ⚙️ Dataset Format
The dataset is structured as follows:

```
├── data
    ├── news
        ├── origin.json
        ├── entity_modifiction.json
        └── writing_style.json
    ├── imgs
        ├── 0.png
        ├── 1.png
        ├── 2.png
        └── ... # {id}.png
```

Format of `news/origin.json`:
```
{
  ID: 1.
  Content: Death of Slim Shady: The controversial legacy of Eminem’s peroxide-blond alter ego...
  Published: 2024-06-01,
  Label: real,
  Evidence: In his new album "The Death of Slim Shady (Coup De Grâce)", Eminem examines the controversial legacy...,
  Evidence Prediction: real,
  Reasoning: The title uses the provocative term "Death" and frames the topic around a "controversial legacy"...,
  Reasoning Prediction: real,
  Sentiment: The title uses metaphorical language common in music journalism...,
  Sentiment Prediction: real,
  Modification Type: origin,
  Sources: BBC,
  Evidence Reliable: 1,
  Reasoning Reliable: 1
},
```

## 👨‍💻 Code

### Environment Setup
1. **Clone the repository:**
```bash 
git clone https://github.com/wgyhhhh/EASE.git
cd EASE
 ```
2. **Install dependencies:**
```bash 
conda create --name EASE python=3.10
conda activate EASE
pip install -r requirements.txt
 ```

### Pretrained BERT

After downloading the pretrained models from their links ([bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)), please configure the local `bert_path` in your scripts.

### Run

#### Training Scripts
```bash
# For training on the Chinese dataset
bash train_zh.sh

# For training on the English dataset
bash train_en.sh
```

#### Testing Scripts
After obtaining the trained weights (saved in `checkpoint_model/{dataset}/Expert_{kind}.pkl`), simply update the corresponding paths in `test.sh` to run batch testing on the news dataset.
```bash
bash test.sh
```

## ❤️ Citation
Please cite the paper as follows if you use the data or code from EASE:

```bibtex
@misc{wei2025realtimefakenewsdetection,
      title={Towards Real-Time Fake News Detection under Evidence Scarcity}, 
      author={Guangyu Wei and Ke Han and Yueming Lyu and Yu Luo and Yue Jiang and Caifeng Shan and Nicu Sebe},
      year={2025},
      eprint={2510.11277},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
