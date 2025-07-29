# 🤖 Sarcasm Detection using LSTM & BERT

Sarcasm — where words say one thing, but mean another. This deep learning project takes a bold step into decoding irony in news headlines using advanced NLP models: **LSTM** and **BERT**.

---

## 📌 Overview

This project focuses on binary classification of news headlines as *sarcastic* or *not sarcastic*, using state-of-the-art Natural Language Processing techniques and deep learning models.

📰 **Dataset**: [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)  
📁 **Notebook**: [`sarcasm_detection.ipynb`](https://github.com/yashika3013/Sarcasm-Detection/blob/main/sarcasm_detection.ipynb)

---

## 🛠️ Tools & Libraries

- Python 🐍
- Pandas, NumPy
- NLTK – for preprocessing
- TensorFlow / Keras – for LSTM model
- Hugging Face Transformers – for BERT model
- Matplotlib, Seaborn – for visualization

---

## 🧠 Model Architectures

### 1. **LSTM (Long Short-Term Memory)**
- Embedding layer + LSTM units
- Ideal for understanding sequential patterns in headlines

### 2. **BERT (Bidirectional Encoder Representations from Transformers)**
- Transformer-based language model
- Captures bidirectional context and semantics

---

## 📊 Results

Experiments were conducted using both **short** and **long** versions of the input headlines. Here's how the models performed:

| Model     | Input Type | Accuracy (%) |
|-----------|------------|--------------|
| BERT      | Long       | 76.65        |
| BERT      | Short      | 83.12        |
| LSTM      | Long       | **87.94** ✅ |
| LSTM      | Short      | 83.86        |

> 🏆 **LSTM with long input length outperformed all**, proving that sequential context pays off.  
> 📈 **BERT** handled short headlines better, suggesting pre-trained embeddings thrive with concise input.

---

## 🚀 Future Scope

- Fine-tune BERT further using more domain-specific data
- Implement attention-based LSTM for improvement
- Build a web app for real-time sarcasm detection
- Explore multilingual sarcasm classification

---


## ⭐ Show Some Love!

If you found this project cool, helpful, or just appropriately sarcastic, consider dropping a ⭐ on the repo. Feedback and PRs are welcome!

---

*“Sure, detecting sarcasm is easy,” said every model before overfitting.* 🙃
