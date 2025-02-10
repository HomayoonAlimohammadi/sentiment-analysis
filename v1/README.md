# Sentiment Analysis with LSTM

A complete implementation of sentiment analysis using LSTM on the IMDB movie review dataset.

## Features
- Text preprocessing with HTML cleaning and lemmatization
- Bidirectional LSTM architecture
- Model evaluation with accuracy metrics and confusion matrix
- Saved model and tokenizer for predictions

## Installation
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn nltk
```

## Usage
1. Run the analysis:
```bash
python main.py
```

2. Expected outputs:
- Trained model (`sentiment_analysis_lstm.h5`)
- Tokenizer (`tokenizer.pkl`)
- Training/validation plots
- Test accuracy (85-88%)

## Example Prediction
```python
sample_review = "This movie was absolutely fantastic!"
print(predict_sentiment(sample_review))
# Output: Sentiment: Positive (Confidence: 0.9783)
```

## Dataset
IMDB Movie Reviews (50,000 labeled reviews)

## Dependencies
- Python 3.7+
- TensorFlow 2.x
- Pandas, Numpy, Matplotlib
- scikit-learn, NLTK

## Model Architecture
- Embedding Layer (128D)
- Bidirectional LSTM (64 units)
- Dense Classifier with Dropout
