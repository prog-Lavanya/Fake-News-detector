# Fake News Detection using Machine Learning

# Project Overview
This project builds a machine learning model that can classify news articles as **Fake** or **Real**.

The dataset contains two CSV files: one with fake news and the other with real news articles. Most of the articles are related to **US political news**.

The goal of this project is to understand the **complete machine learning workflow**, starting from data preparation to model evaluation and analysis.

---

# Basic Machine Learning Concepts

## 1. Overfitting
Overfitting happens when the model **memorizes the training data instead of learning general patterns**.

### Characteristics
- The model becomes too complex
- Very high accuracy on training data
- Poor performance on unseen data

Example pattern:

```
High training accuracy
Low testing accuracy
```

This means the model **cannot generalize well to new data**.

---

## 2. Underfitting
Underfitting occurs when the model is **too simple to capture the patterns in the dataset**.

### Characteristics
- The model fails to learn relationships in the data
- Accuracy is low

Example pattern:

```
Low training accuracy
Low testing accuracy
```

This means the model is **not learning enough from the data**.

---

# Dataset Used

Two CSV files are used in this project:

- Fake.csv
- True.csv

Both datasets contain news articles with:

- Title
- News content

During preprocessing, both datasets are merged and a **label column** is created:

```
Fake News → 0
Real News → 1
```

---

# Purpose of Train/Test Split

The dataset is divided into two parts:

### Training Data
Used to teach the model patterns in the data.

### Testing Data
Used to evaluate how well the model performs on **new unseen data**.

Simple analogy:

```
Training → Practice
Testing → Final Exam
```

---

# Why Text Cannot Be Used Directly

Machine learning models **cannot understand raw text**.  
They only work with **numerical values**.

Therefore, text must first be converted into numbers using a process called **text vectorization**.

---

# NLP Preprocessing Pipeline

Before training the model, text data goes through several preprocessing steps:

```
Raw Text
   ↓
Cleaning
   ↓
Tokenization
   ↓
Remove Stopwords
   ↓
Vectorization (TF-IDF)
   ↓
Machine Learning Model
```

---

# Tokenization
Tokenization means **splitting text into individual words**.

Example:

```
"Machine learning is powerful"
```

becomes

```
["machine", "learning", "is", "powerful"]
```

---

# TF-IDF Vectorization

- TF-IDF converts text into **numerical features**.
- It gives higher importance to **important words** and lower importance to very common words.

This allows machine learning models to **process textual data effectively**.

---

# Models Used

Two classification models were trained and compared:

1. Logistic Regression
2. Naive Bayes

---

# Logistic Regression

Logistic Regression is a **classification algorithm**.

It predicts probabilities and then assigns a class.

Example process:

```
Input text → Probability calculation → Fake or Real
```

If:

```
P(Fake) > 0.5
```

then the model predicts **Fake News**.

Otherwise it predicts **Real News**.

Logistic Regression uses the **sigmoid function**:

```
P = 1 / (1 + e^(-z))
```

---

# Naive Bayes

Naive Bayes assumes that **all words are independent of each other**.

Because of this assumption, it sometimes produces slightly lower accuracy.

However, it has several advantages:

- Very fast
- Works well on small datasets
- Uses low memory

---

# Why Accuracy Alone Is Not Enough

Accuracy only tells us **how many predictions were correct overall**, but it does not explain:

- Which class was misclassified
- Whether the model is biased
- How many important cases were missed

Therefore, additional evaluation metrics are used.

---

# Confusion Matrix

A confusion matrix compares **actual values with predicted values**.

|                | Predicted Fake | Predicted Real |
|---------------|---------------|---------------|
| Actual Fake   | True Positive | False Negative |
| Actual Real   | False Positive | True Negative |

This helps us understand **where the model is making mistakes**.

---

# Precision

Precision measures the **quality of positive predictions**.

Formula:

```
Precision = True Positives / (True Positives + False Positives)
```

Meaning:

Out of all items predicted as positive, **how many were actually correct**.

---

# Recall

Recall measures **how many actual positive cases the model successfully detected**.

Formula:

```
Recall = True Positives / (True Positives + False Negatives)
```

Recall is also known as:

- Sensitivity
- True Positive Rate

It focuses on **not missing important cases**.

---

# Precision–Recall Tradeoff

Changing the classification threshold affects precision and recall.

```
Lower threshold → Higher Recall → Lower Precision
Higher threshold → Lower Recall → Higher Precision
```

---

# Model Training Process

The training pipeline used in this project:

```
Dataset
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
```

During training, the model learns **weights for different words and patterns** in the dataset.

These learned patterns help the model classify future news articles.

---

# Model Performance

Two models were trained and compared:

```
Logistic Regression Accuracy: 0.9860801781737194
Naive Bayes Accuracy: 0.9338530066815145
```

Logistic Regression performed better and was selected as the final model.

---

# Project Structure

```
fake-news-detector
│
├── Fake.csv
├── True.csv
├── main.py
├── fake_news_model.pkl
├── vectorizer.pkl
└── README.md
```

---

# Key Learnings

Through this project the following concepts were understood:

- Complete machine learning workflow
- Text preprocessing for NLP tasks
- TF-IDF feature extraction
- Training and comparing classification models
- Evaluating models using multiple metrics
- Understanding model behavior using confusion matrix and error analysis
