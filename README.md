# 🕵️ Fake Review Detection using Machine Learning

This project aims to detect fake (spam) product reviews using Natural Language Processing (NLP) and machine learning classification algorithms. It analyzes review text and classifies whether it is genuine or deceptive based on patterns learned from the data.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Setup & Usage](#setup--usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 📌 Features

- 🔍 Detects fake vs genuine product reviews using supervised ML
- 🧠 NLP processing with TF-IDF vectorization
- 🏗️ Clean architecture with modular Python scripts
- 📈 Metrics: Accuracy, Precision, Recall, F1-Score
- 💾 Saved trained model using `joblib` and `pickle`
- 🎛️ Simple text-based interface to test predictions

---

## 🧰 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- PyCharm (or any Python IDE)

---

## 📂 Dataset

- The dataset consists of product reviews labeled as either **deceptive** or **truthful**
- Preprocessing is done using NLP techniques like:
  - Tokenization
  - Stopword removal
  - TF-IDF vectorization

> Source: Synthetic/spam review datasets (Ott et al. – Cornell)

---

## 🚀 Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone git@github.com:parth4673/fake-review-detector.git
   cd fake-review-detector
