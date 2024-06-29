# Sentiment Analysis and Customer Segmentation Project

This project consists of two main tasks: Sentiment Analysis on movie reviews and Customer Segmentation based on purchase behavior.

## Introduction
The goal of this project is to:
1. Perform sentiment analysis on movie reviews using Natural Language Processing (NLP) techniques.
2. Apply K-Means clustering to segment customers based on their purchase behavior.

## Technologies Used
- Python
- Pandas
- NLTK
- SpaCy
- Scikit-Learn
- Matplotlib

## Dataset
[imdb](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
### Movie Reviews Dataset
- **Source**: IMDb
- **Columns**: `review`, `sentiment`
- **Description**: Contains movie reviews and their corresponding sentiments (positive/negative).

### Customer Purchase Behavior Dataset
[onlineretail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Source**: Online Retail
- **Columns**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- **Description**: Contains transactional data of an online retail store.

## Sentiment Analysis on Movie Reviews
### Steps:
1. **Loading the Dataset**: Load the IMDb movie reviews dataset.
2. **Preprocessing Text**: Clean and preprocess the text data using SpaCy and NLTK.
3. **Vectorization**: Convert text data into numerical format using TF-IDF Vectorizer.
4. **Model Training**: Train a Logistic Regression model to predict sentiments.
5. **Model Evaluation**: Evaluate the model using precision, recall, and F1-score.

## Results
**Sentiment Analysis:**
Accuracy: 88%
Precision, Recall, F1-Score: Balanced performance across positive and negative sentiments.
**Customer Segmentation:**
Cluster 0: Low spenders, infrequent and inactive.
Cluster 1: High spenders, very frequent and active.
Cluster 2: Moderate spenders, moderately frequent and active.
