import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# 1. Load dataset
data = pd.read_csv("student_scores.csv")

X = data[["Hours"]]
Y = data["Score"]

# 2. Create deep learning model
model = Sequential([
    Input(shape=(1,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

# 3. Compile model with a slightly higher learning rate for faster convergence on small data
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# 4. Train model for more epochs to ensure proper fitting
model.fit(X, Y, epochs=1000)

# 5. Predict score using a NumPy array
prediction_input = np.array([[7.0]])
prediction = model.predict(prediction_input)
print("Predicted Score for 7.0 hours:", prediction[0][0])

# 6. Plot graph
plt.scatter(X, Y, label="Actual", color='blue')
plt.plot(X, model.predict(X), color='red', label="Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Deep Learning - Score Prediction")
plt.legend()
plt.show()