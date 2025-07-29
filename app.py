from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize model variables
model_short = None
model_long = None

def load_models():
    global model_short, model_long

    vocab_size = len(tokenizer.word_index) + 1

    # Load short headlines model
    model_short = Sequential()
    model_short.add(Embedding(
        input_dim=vocab_size,
        output_dim=200,  # Assuming the dimension of your embeddings
        mask_zero=True
    ))
    model_short.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.5, dropout=0.5)))
    model_short.add(Dense(1, activation='sigmoid'))

    model_short.build((None, 25))  # Assuming input shape is (None, 25)

    try:
        # Load short model weights
        model_short.load_weights('model_s.weights.h5')
    except ValueError as e:
        print("Error loading weights:", e)

    # Load long headlines model
    model_long = Sequential()
    model_long.add(Embedding(
        input_dim=vocab_size,
        output_dim=200,  # Assuming the dimension of your embeddings
        mask_zero=True
    ))
    model_long.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.5, dropout=0.5)))
    model_long.add(Dense(1, activation='sigmoid'))

    # Build the long model before loading weights
    model_long.build((None, 25))  # Assuming input shape is (None, 25)

    try:
        # Load long model weights
        model_long.load_weights('model_l.weights.h5')
    except ValueError as e:
        print("Error loading weights:", e)

load_models()

# Function to preprocess input statements and make predictions
def preprocess_statement(statement, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([statement])
    padded_sequence = pad_sequences(sequence, padding='pre', maxlen=25)
    prediction = model.predict(padded_sequence)
    sarcasm_label = "Sarcastic" if prediction > 0.5 else "Not Sarcastic"
    return sarcasm_label, statement

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    statement = request.form['statement']
    length = len(statement.split())  # Calculate the length of the input statement
    if length <= 10:  # Example threshold for short sentences
        model = model_short
    else:
        model = model_long

    sarcasm_label, statement = preprocess_statement(statement, tokenizer, model)
    return render_template('index.html', sarcasm_label=sarcasm_label, statement=statement)

if __name__ == '__main__':
    app.run(debug=True)
