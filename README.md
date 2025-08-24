
# 🚀 Character-Level Language Models with PyTorch  

This repository contains two implementations of character-level language models built using **PyTorch**:  

1. **Bigram Model** (`bigram.py`) → A simple baseline language model based on bigram probabilities.  
2. **GPT-Like Transformer Model** (`gpt.py`) → A scaled-down implementation of a GPT-style Transformer for better performance.  

The models are trained on the **Tiny Shakespeare dataset** and can generate new text sequences resembling Shakespearean writing.  

---

## **📌 Project Structure**
```
.
├── bigram.py      # Simple bigram-based language model
├── gpt.py         # GPT-like Transformer-based language model
├── input.txt      # Training dataset (Tiny Shakespeare)
├── more.txt       # Example generated output from GPT
└── README.md      # Project documentation
```

---

## **⚡ Features**
- Implements two different character-level models:
  - **Bigram Model** → Very basic, predicts next character based on the current one.
  - **Transformer GPT Model** → Uses multi-head self-attention, positional embeddings, and feed-forward layers.
- Trains on any custom text dataset (`input.txt` by default).
- Generates **novel Shakespeare-like text**.
- Uses **PyTorch** for model definition and training.
- GPU acceleration supported (`CUDA` automatically enabled if available).

---

## **🛠 Installation**

Clone this repository and install dependencies:  

```bash
git clone https://github.com/yourusername/char-level-transformer.git
cd char-level-transformer
pip install torch numpy
```

> **Optional:** If you want to use a GPU, make sure to install the CUDA-enabled PyTorch version from [pytorch.org](https://pytorch.org/).

---

## **📂 Dataset**
By default, the models are trained on the **Tiny Shakespeare dataset**:  

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Place `input.txt` in the project root directory.

---

## **🚀 Usage**

### **1. Train & Generate with Bigram Model**
Run:
```bash
python bigram.py
```
This will:
- Train a bigram-based language model.
- Print training & validation losses.
- Generate a sample of **500 characters** and display it.

---

### **2. Train & Generate with GPT Transformer Model**
Run:
```bash
python gpt.py
```
This will:
- Train a GPT-like Transformer model.
- Show step-wise training and validation loss.
- Generate a Shakespeare-style passage of **500 characters**.
- Optionally, save large generated outputs into `more.txt`.

---

## **🔧 Model Details**

### **Bigram Model** (`bigram.py`)
- Uses a single **embedding layer** to learn a probability distribution over the next character.
- Very small and fast to train.
- Serves as a baseline for performance comparison.

### **GPT-Like Transformer** (`gpt.py`)
- Implements a mini **GPT-style Transformer** with:
  - **Multi-Head Self-Attention**
  - **Positional Embeddings**
  - **Feed-Forward Networks**
  - **Layer Normalization & Dropout**
- Configurable hyperparameters:
    - `n_embd = 384` → Embedding dimension
    - `n_head = 6` → Number of attention heads
    - `n_layer = 6` → Number of Transformer blocks
    - `block_size = 256` → Context size
    - `batch_size = 64`
    - `learning_rate = 3e-4`
- Produces **much better results** compared to the Bigram model.

---

## **📊 Example Output**

Example from **more.txt** after training the GPT model:  

```
LUCIO:
We muse hath resistes him so sovere: son't his other wrough
stands of coverent sh'd: he has here, and stand it
and poor exceeder or a Henry's last, stay
not in faith, forewell's base of graves, thanks, happy comparel...
```

The generated text mimics Shakespeare’s style but is **entirely machine-generated**.

---

## **📈 Training Performance**
| Model      | Parameters | Training Speed | Text Quality |
|-----------|------------|---------------|--------------|
| Bigram    | ~30K      | Very Fast ⚡   | Poor 🟡 |
| GPT       | ~10M      | Slower ⏳      | Much Better 🟢 |

---

## **📌 Future Improvements**
- [ ] Add **beam search** for better generation quality.
- [ ] Support **token-level GPT** instead of character-level.
- [ ] Add support for **custom datasets**.
- [ ] Include pretrained weights for quick testing.
