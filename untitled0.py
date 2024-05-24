# -*- coding: utf-8 -*-
"""
Created on Fri May 24 07:23:44 2024

@author: UseR
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Function to clean text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None)
data.columns = ['review', 'sentiment']
print("Cleaning text data...")
data['review'] = data['review'].apply(clean_text)

# Tokenize and pad sequences
print("Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['review'])
X = tokenizer.texts_to_sequences(data['review'])
X = pad_sequences(X, maxlen=100)
y = np.array(data['sentiment'])

# Split the data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
print("Building the model...")
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
print("Compiling the model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {accuracy}')

# Make predictions
sample_review = ["Just killin it."]
sample_sequence = tokenizer.texts_to_sequences(sample_review)
sample_padded = pad_sequences(sample_sequence, maxlen=100)
prediction = model.predict(sample_padded)
print(f'Sentiment: {"Positive" if prediction[0][0] > 0.5 else "Negative"}')
