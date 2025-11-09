# Sentiment Analysis: Logistic Regression Baseline

A classical machine learning approach to toxic tweet classification using Logistic Regression with TF-IDF vectorization, establishing baseline performance for multi-class sentiment analysis on social media text.

## 🎯 Project Overview

This project implements a traditional machine learning pipeline for classifying tweets into three sentiment categories. Using Logistic Regression with TF-IDF features, this baseline model demonstrates fundamental NLP techniques and serves as a performance benchmark for more complex deep learning architectures.

**Classification Task:**
- **Class 0:** Hate Speech
- **Class 1:** Offensive Language
- **Class 2:** Neutral

**Objective:** Build an interpretable, efficient baseline model for offensive content detection on social media platforms.

## 🛠️ Technical Architecture

### Machine Learning Pipeline

**Complete Workflow:**
```
Raw Tweets
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Feature Standardization
    ↓
Logistic Regression
    ↓
Multi-class Predictions
```

### Text Preprocessing

**Cleaning Steps:**
- URL removal (http/https patterns)
- Mention removal (@username)
- Special character filtering
- Lowercase normalization
- Tokenization
- Stopword removal (optional based on experimentation)

### Feature Extraction

**TF-IDF Vectorization:**
- **Method:** Term Frequency-Inverse Document Frequency
- **Why TF-IDF:** Captures word importance while reducing common word weight
- **Vocabulary:** Most frequent terms across corpus
- **N-grams:** Unigrams and bigrams for context capture
- **Max Features:** Limited vocabulary size for efficiency

**Advantages:**
- Interpretable feature weights
- Handles sparse text data effectively
- Fast computation and inference
- Works well with linear models

### Feature Standardization

**Standard Scaler:**
- Normalizes TF-IDF features to zero mean, unit variance
- Improves logistic regression convergence
- Prevents feature scale bias
- Saved for consistent test set transformation

### Model Architecture

**Logistic Regression:**
```python
Model Configuration:
├── Multi-class: One-vs-Rest (OvR) strategy
├── Regularization: L2 penalty
├── Solver: Optimized for multi-class
├── Max iterations: Tuned for convergence
└── Class weights: Balanced for imbalanced data
```

**Why Logistic Regression:**
- Fast training and inference
- Interpretable coefficients (feature importance)
- Probabilistic predictions
- Strong baseline for text classification
- Handles high-dimensional sparse features well

## 💡 Implementation Details

### Training Configuration

**Data Split:**
- Training/Validation split for model evaluation
- Stratified sampling to maintain class distribution
- Cross-validation for hyperparameter tuning (optional)

**Hyperparameter Tuning:**
- Regularization strength (C parameter)
- TF-IDF max features
- N-gram range selection
- Class weight balancing

**Optimization:**
- Convergence tolerance tuning
- Iteration limit adjustment
- Multi-class strategy selection

### Model Persistence

**Saved Artifacts:**
- `logistic_regression_weights.joblib` - Trained model weights
- `standard_scaler.joblib` - Feature scaler for test set
- Enables reproducible predictions without retraining

## 📊 Evaluation Metrics

**Performance Metrics:**
- **Accuracy:** Overall classification correctness
- **Precision (macro):** Average precision across classes
- **Recall (macro):** Average recall across classes
- **F1 Score (macro):** Harmonic mean of precision and recall

**Why Macro Averaging:**
- Treats all classes equally (fair for imbalanced data)
- Prevents bias toward majority class
- Important for minority hate speech class detection

## 🚀 Skills Demonstrated

**Classical Machine Learning:**
- Logistic regression for multi-class classification
- TF-IDF vectorization for text representation
- Feature engineering and selection
- Regularization techniques

**Natural Language Processing:**
- Text preprocessing and normalization
- Tokenization strategies
- Stopword handling
- N-gram feature extraction

**ML Engineering:**
- Model serialization and persistence
- Feature scaling and standardization
- Hyperparameter tuning
- Cross-validation methodology

**Software Engineering:**
- Reproducible ML pipelines
- Model versioning with joblib
- Clean code organization
- Kaggle competition workflow

## 📁 Project Structure
```
sentiment-analysis-baseline/
├── CODE.ipynb                          # Main notebook
├── logistic_regression_weights.joblib  # Trained model
├── standard_scaler.joblib              # Feature scaler
├── mimisubmission.csv                  # Kaggle predictions
└── README.md                           # This file
```

## 🎯 Results & Insights

**Model Performance:**
- **Grade:** 70/100 (Kaggle competition scoring)
- Competitive baseline for linear models
- Room for improvement with deep learning approaches

**Key Findings:**
- TF-IDF captures important toxic keywords effectively
- Logistic regression provides interpretable feature weights
- Linear models struggle with complex contextual patterns
- Motivates exploration of neural architectures (RNNs, CNNs)

**Baseline Importance:**
- Establishes minimum performance threshold
- Fast iteration for feature engineering experiments
- Interpretable for understanding important words
- Production-ready for real-time inference

## 🔧 Technical Insights

### Logistic Regression Advantages

**For Text Classification:**
- Naturally handles sparse TF-IDF features
- Fast training on large vocabularies
- Interpretable coefficients show toxic word importance
- Probabilistic outputs for confidence thresholds

**Production Considerations:**
- Low latency inference (milliseconds)
- Small model size (easy deployment)
- No GPU requirements
- Simple monitoring and debugging

### Limitations & Future Work

**Current Limitations:**
- Cannot capture word order semantics
- Misses contextual nuances and sarcasm
- Bag-of-words ignores sentence structure
- Limited handling of rare words/misspellings

**Future Enhancements:**
- Word embeddings (Word2Vec, GloVe)
- Recurrent neural networks (LSTM, GRU)
- Contextualized embeddings (BERT)
- Ensemble with deep learning models

## 🌍 Real-World Applications

**Content Moderation:**
- Fast pre-filtering for human review
- Real-time API for social platforms
- Mobile app integration (lightweight)
- Edge deployment on low-resource devices

**Baseline Benchmarking:**
- Performance reference for complex models
- Cost-benefit analysis (accuracy vs. complexity)
- A/B testing against neural networks
- Fallback model for production systems

## 📄 Competition Details

**Kaggle Competition:** Sentiment Analysis - Toxic Tweet Classification

**Submission:**
- Format: CSV with tweet IDs and predicted classes
- Evaluation: Macro F1 score on held-out test set
- Leaderboard: Competitive ranking among peers

**Learning Outcomes:**
- Kaggle competition workflow
- Model submission and evaluation
- Baseline establishment methodology
- Iterative improvement process

## 🎓 Project Context

Developed as the foundational assignment in a machine learning course, this project establishes classical ML proficiency before advancing to deep learning techniques. The logistic regression baseline demonstrates understanding of fundamental NLP pipelines, feature engineering, and model evaluation - essential skills for any ML practitioner.

This work serves as the performance benchmark for subsequent assignments exploring neural architectures (RNNs, LSTMs, GRUs, CNNs), highlighting the trade-offs between model complexity, interpretability, and accuracy in production NLP systems.

## 🔗 Connect

**Meriem Lmoubariki**
- GitHub: [@myriamlmiii](https://github.com/myriamlmiii)

---

*Building interpretable baselines for responsible AI and content moderation.*
