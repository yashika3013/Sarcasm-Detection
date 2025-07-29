import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import random
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle
# Load the data
file_path = 'Sarcasm_Headlines_Dataset_v2.json'
df = pd.read_json(file_path, lines=True)
print(df.head())

# Preprocess the data
df.drop(columns='article_link', axis=1, inplace=True)
df = df.drop_duplicates('headline')

tokenize = Tokenizer(oov_token='<oov>')  # creating keras tokenizer object
tokenize.fit_on_texts(df['headline'])  # build the vocabulary
word_index = tokenize.word_index  # mapping words to their respective int indices from fitted tokenizer
df['headline_sequence'] = tokenize.texts_to_sequences(df['headline'])  # convert headlines into integer sequences using the word index
df['length'] = df['headline_sequence'].apply(len)  # calculate the number of words in headlines

df = df.sort_values(by='length')

df_clean = df[df['length'] <= 25]
print(df_clean.groupby('is_sarcastic').describe())

df_short = df_clean[df_clean['length'] <= df_clean['length'].median()]
df_long = df_clean[df_clean['length'] > df_clean['length'].median()]

print(df_short.groupby('is_sarcastic').describe(), df_long.groupby('is_sarcastic').describe())

# Prepare the data for model training
X_long = df_long['headline']
label_long = to_categorical(df_long['is_sarcastic'], 2)

X_short = df_short['headline']
label_short = to_categorical(df_short['is_sarcastic'], 2)

Y_long = df_long['is_sarcastic']
Y_short = df_short['is_sarcastic']

x_long_padded = pad_sequences(df_long['headline_sequence'], padding='pre')
X_long_train, X_long_test, Y_long_train, Y_long_test = train_test_split(
    x_long_padded,
    Y_long,
    random_state=0,
    stratify=Y_long  # ensures that class distribution is maintained in splits
)

x_short_padded = pad_sequences(df_short['headline_sequence'], padding='pre')
X_short_train, X_short_test, Y_short_train, Y_short_test = train_test_split(
    x_short_padded,
    Y_short,
    random_state=0,
    stratify=Y_short  # ensures that class distribution is maintained in splits
)

# Load GloVe embeddings and create embedding matrix
emb_file = 'glove.twitter.27B.200d.txt'  # Path to your GloVe embeddings file
embedding_dim = 200  # Assuming the dimension of your GloVe embeddings is 200

# Load GloVe embeddings into a dictionary
embeddings_index = {}
with open(emb_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Check embedding matrix shape
expected_shape = (len(word_index) + 1, embedding_dim)
if embedding_matrix.shape != expected_shape:
    raise ValueError(f"Embedding matrix shape mismatch. Expected {expected_shape}, got {embedding_matrix.shape}")

# Building Models
random.seed(2023)
model_long = Sequential()
model_long.add(Embedding(
    input_dim=len(word_index) + 1,
    output_dim=embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),  # Initialize embeddings with the pre-trained weights
    trainable=True,  
    mask_zero=True  # If your input sequences have varying lengths and you want to mask padding
))
model_long.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.5, dropout=0.5)))
model_long.add(Dense(1, activation='sigmoid'))  # binary classification

model_long.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['acc']
)

# Train the model
history_long = model_long.fit(
    X_long_train,
    Y_long_train,
    batch_size=128,
    validation_data=(X_long_test, Y_long_test),
    epochs=2 
)

lstm_long_val_acc = round(history_long.history['val_acc'][-1] * 100, 2)

# Print the final validation accuracy
print(lstm_long_val_acc)

# Building Models for Short Headlines
random.seed(2023)
model_short = Sequential()
model_short.add(Embedding(
    input_dim=len(word_index) + 1,
    output_dim=embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True,
    mask_zero=True
))
model_short.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.5, dropout=0.5)))
model_short.add(Dense(1, activation='sigmoid'))  # binary classification

model_short.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['acc']
)

# Train the short headlines model
history_short = model_short.fit(
    X_short_train,
    Y_short_train,
    batch_size=128,
    validation_data=(X_short_test, Y_short_test),
    epochs=2
)

# Calculate and print the validation accuracy for the short headlines model
lstm_short_val_acc = round(history_short.history['val_acc'][-1] * 100, 2)
print(lstm_short_val_acc)

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_short.save_weights('model_s.weights.h5')
model_long.save_weights('model_l.weights.h5')
