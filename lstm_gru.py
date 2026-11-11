
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU,Embedding,Dense
import matplotlib.pyplot as plt

data=pd.read_csv("dataset\IMDB\IMDB Dataset.csv")

x=data["review"].values
y = data["sentiment"].map(lambda x: 1 if x == "positive" else 0).values

tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x)
x=tokenizer.texts_to_sequences(x)

x=pad_sequences(x,maxlen=100)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

layer=[SimpleRNN,LSTM,GRU]
accu=[]

for laye in layer:
    model=Sequential([Embedding(10000,32),laye(32),Dense(1,activation="sigmoid")])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    start=time.time()
    model.fit(x_train,y_train,epochs=10,batch_size=64)
    loss,accuracy=model.evaluate(x_test,y_test)
    accu.append(accuracy)
    print(accuracy)

plt.bar(["SimpleRNN","LSTM","GRU"], accu)
plt.show()


# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import time

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# # 1. Load CSV dataset
# data = pd.read_csv("dataset/IMDB/IMDB Dataset.csv")

# X = data['review'].values
# y = data['sentiment'].map({'positive': 1, 'negative': 0}).values

# # 2. Tokenization
# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(X)

# X = tokenizer.texts_to_sequences(X)
# X = pad_sequences(X, maxlen=100)

# # 3. Split data
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 4. Define models
# names = ['RNN', 'LSTM', 'GRU']
# layers = [SimpleRNN, LSTM, GRU]

# accuracies = []

# # 5. Train and test
# for name, layer in zip(names, layers):

#     print("\nTraining", name)

#     model = Sequential([
#         Embedding(10000, 32),
#         layer(32),
#         Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     start = time.time()

#     model.fit(X_train, y_train, epochs=3, batch_size=128)

#     loss, accuracy = model.evaluate(X_test, y_test)

#     accuracies.append(accuracy * 100)

#     print(name, "Accuracy:", accuracy * 100, "%")
#     print("Time:", round(time.time() - start, 2), "seconds")

# # 6. Compare accuracy
# plt.bar(names, accuracies)
# plt.title("RNN vs LSTM vs GRU")
# plt.xlabel("Model")
# plt.ylabel("Accuracy (%)")
# plt.ylim(0, 100)
# plt.savefig("rnn_lstm_gru_comparison.png")
# print("Saved comparison chart to rnn_lstm_gru_comparison.png")
# plt.show()