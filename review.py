
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Load IMDB dataset from CSV
data = pd.read_csv("dataset/IMDB/IMDB Dataset.csv")

# Extract reviews and sentiments
texts = data["review"].values
y = data["sentiment"].map(lambda x: 1 if x == "positive" else 0).values

# 2. Tokenize words and pad reviews
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=200)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define LSTM model
model = Sequential([
    Embedding(10000, 64),
    LSTM(64),
    Dense(2, activation='softmax')
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
model.fit(X_train, y_train, epochs=3, batch_size=64)

# 6. Test model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy * 100, "%")

# 7. Predict user review
def predict_review(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded, verbose=0).argmax()

    if prediction == 1:
        print("Sentiment: Positive 😊")
    else:
        print("Sentiment: Negative 😞")

# Enter review
review = input("Enter a movie review: ")
predict_review(review)
