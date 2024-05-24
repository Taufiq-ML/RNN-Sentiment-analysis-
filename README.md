
# IMDB Movie Reviews Sentiment Analysis

This repository contains a Python script for performing sentiment analysis on IMDB movie reviews using a Recurrent Neural Network (RNN) with an LSTM layer. The script preprocesses the text data, builds and trains the model, and evaluates its performance.

## Dataset

The dataset used is `imdb_labelled.txt`, which consists of movie reviews and their corresponding sentiments (0 for negative and 1 for positive).

## Requirements

- Python 3.x
- pandas
- numpy
- nltk
- tensorflow
- scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Taufiq-ML/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2. Install the required packages:
    ```sh
    pip install pandas numpy nltk tensorflow scikit-learn
    ```

3. Download NLTK stopwords:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

1. Ensure you have the `imdb_labelled.txt` file in the same directory as the script.
2. Run the script:
    ```sh
    python sentiment_analysis.py
    ```

## Script Overview

- **Data Loading and Cleaning**:
    - Load the dataset using pandas.
    - Clean the text data by removing HTML tags, non-alphabetical characters, converting to lowercase, and removing stopwords.
  
- **Tokenization and Padding**:
    - Tokenize the text data and convert it into sequences.
    - Pad the sequences to ensure uniform length.

- **Model Building**:
    - Build a Sequential model using Keras with an Embedding layer, LSTM layer, Dropout layer, and a Dense output layer.

- **Training and Evaluation**:
    - Compile the model with Adam optimizer and binary cross-entropy loss.
    - Train the model and evaluate its performance on a test set.
  
- **Making Predictions**:
    - Use the trained model to predict the sentiment of sample reviews.

## Example Output

```
Num GPUs Available:  1
Loading dataset...
Cleaning text data...
Tokenizing and padding sequences...
Splitting data into training and test sets...
Building the model...
Compiling the model...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 100, 128)          640000
_________________________________________________________________
lstm (LSTM)                  (None, 32)                20608
_________________________________________________________________
dropout (Dropout)            (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 660,641
Trainable params: 660,641
Non-trainable params: 0
_________________________________________________________________
Training the model...
Epoch 100/100
Epoch 97/100
19/19 - 0s - 22ms/step - accuracy: 1.0000 - loss: 5.6306e-04 - val_accuracy: 0.7667 - val_loss: 1.4543
Epoch 98/100
19/19 - 0s - 24ms/step - accuracy: 1.0000 - loss: 9.6709e-04 - val_accuracy: 0.7667 - val_loss: 1.4433
Epoch 99/100
19/19 - 0s - 24ms/step - accuracy: 1.0000 - loss: 6.5609e-04 - val_accuracy: 0.7667 - val_loss: 1.4387
Epoch 100/100
19/19 - 0s - 23ms/step - accuracy: 1.0000 - loss: 4.8037e-04 - val_accuracy: 0.7667 - val_loss: 1.4504
Evaluating the model...
5/5 - 0s - 6ms/step - accuracy: 0.7667 - loss: 1.4504
Test Accuracy: 0.7666666507720947
Sentiment: Positive
```

## Contributing

If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


