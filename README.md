Here’s the **edited and professional version of the README** that matches your implementation with `50 epochs`:

---

# 📚 Sentiment Analysis of Movie Reviews using Word2Vec & LSTM (PyTorch)

This project implements a **Sentiment Analysis** system for movie reviews using a **Long Short-Term Memory (LSTM)** network. The model is trained to classify each review as either **positive** or **negative**, leveraging **Word2Vec embeddings** for word representation and **PyTorch** for deep learning.

---

## 🔧 Features & Pipeline

1. **Data Loading & Preprocessing**

   * Reads raw review data from `reviews.txt` and corresponding sentiment labels from `labels.txt`.
   * Tokenizes, encodes, and pads each review to a fixed length.

2. **Word2Vec Embedding**

   * Uses learned Word2Vec-style embeddings (400 dimensions) to convert words into dense vectors.

3. **LSTM Network**

   * A deep LSTM network captures sequential dependencies in review text, leveraging multiple LSTM layers.

4. **Model Training & Evaluation**

   * The model is trained using Binary Cross-Entropy Loss and evaluated on a test set.
   * Supports saving and loading model checkpoints during training.

5. **Prediction**

   * Given any new movie review, the model returns:

     * Cleaned tokens
     * Encoded tensor
     * Raw sentiment score (0 to 1)
     * Final predicted label (positive/negative)

---

## 🧠 Model Architecture

### I. **Word2Vec Embedding Layer**

* Converts input text to 400-dimensional dense vectors.
* Reduces dimensionality while preserving semantic similarity between words.

### II. **LSTM Layers**

* Stacked LSTM layers (2 deep) with hidden size of 256.
* Learns long-term dependencies in sequential data.

---

## 📁 Repository Contents

| File / Folder                | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| `sentiment_analysis_LSTM.py` | Main script to load data, train, evaluate and predict using the model |
| `data/reviews.txt`           | Contains one movie review per line                                    |
| `data/labels.txt`            | Contains labels (`positive` / `negative`) for each review             |

---

## 📌 Hyperparameters Used

| Hyperparameter      | Value     |
| ------------------- | --------- |
| Batch Size          | **50**    |
| Sequence Length     | **200**   |
| Embedding Dimension | **400**   |
| LSTM Hidden Size    | **256**   |
| LSTM Layers         | **2**     |
| Learning Rate       | **0.001** |
| Gradient Clipping   | **5**     |
| Number of Epochs    | **50**    |

---

## ✅ Example Output

```text
📝 Tokenized: ['this', 'movie', 'was', 'absolutely', 'wonderful']
🔢 Encoded: tensor([[ 319,  636,  413, 1865, 1621]], device='cuda:0')
📈 Raw Score: 0.9121
🎯 Prediction: positive
```

---

## 📚 References

* [Udacity Deep Learning with PyTorch](https://github.com/udacity/deep-learning-v2-pytorch)
* PyTorch Documentation
* Sentiment classification examples on IMDB dataset

---

Let me know if you'd like a badge section (e.g., Python version, PyTorch version), usage guide, or Colab support section added too.
