# Twitter Sentiment Analysis

## Overview

This project implements a Twitter Sentiment Analysis system that classifies tweets as either positive or negative based on their content. The dataset used for training and testing is the Sentiment140 dataset from Kaggle. The project applies Natural Language Processing (NLP) techniques for text preprocessing and employs a Logistic Regression model for classification.

## Dataset

* **Source**: [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
  
* **Size**: 1.6 million tweets
  
* **Labels**:
    * 0: Negative Sentiment
    * 4: Positive Sentiment (Converted to 1 in preprocessing)
 
* **Columns**:
    * `target`: Sentiment label (0 or 1)
    * `id`: Unique tweet ID
    * `date`: Timestamp of the tweet
    * `flag`: Annotation flag
    * `user`: Username of the tweet author
    * `text`: Actual tweet content

## Files

* `twitter_sentiment_analysis.ipynb`: Jupyter Notebook implementing the full analysis pipeline.
* `trained_model.sav`: Saved logistic regression model for reusability.

## Requirements

Install the necessary dependencies using:

    ```
    pip install kaggle pandas numpy nltk scikit-learn
    ```

## Steps Implemented

### 1. Data Preprocessing

**Stemming**: Reducing words to their root forms using PorterStemmer.

**Stopword Removal**: Removing common words that do not add meaning.

**Text Cleaning**: Removing non-alphabetic characters.

**Label Conversion**: Changing 4 labels to 1.

### 2. Feature Extraction

Converting text data into numerical form using TF-IDF Vectorization.

### 3. Splitting Data

The dataset is split into training (80%) and testing (20%) sets.

### 4. Model Training

Logistic Regression is used as the classification model.

The model is trained on the TF-IDF-transformed data.

### 5. Model Evaluation

Accuracy score on training data: **80.2%**

Accuracy score on test data: **77.6%**

### 6. Saving & Reusing the Model

The trained model is saved using pickle.

The model can be reloaded and used for predicting sentiment on new tweets.

Usage

Running the Model on a Sample Tweet

    ```
    import pickle
    
    # Load the trained model
    model = pickle.load(open('trained_model.sav', 'rb'))
    
    # Predict sentiment for a new tweet
    new_tweet = ["I love this product! It's amazing!"]
    prediction = model.predict(new_tweet)
    
    if prediction[0] == 0:
        print("Negative Tweet")
    else:
        print("Positive Tweet")
    ```

## Conclusion

This project successfully classifies tweets based on sentiment using Logistic Regression and NLP techniques. The trained model achieves a reasonable accuracy and can be used for real-time sentiment analysis applications.

## Future Improvements

Implement Deep Learning models (LSTMs, Transformers) for better accuracy.

Enhance preprocessing by handling emoji-based sentiments.

Deploy as a web application for real-time analysis.

Author

Deepak


