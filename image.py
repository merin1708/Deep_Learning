import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from torchvision.datasets import MNIST

# Load directly from the exam folder (100% offline)
train = MNIST(root="dataset/MNIST/data", train=True, download=False)
test = MNIST(root="dataset/MNIST/data", train=False, download=False)

x_train, y_train = train.data, np.array(train.targets)
x_test, y_test = test.data, np.array(test.targets)

# 2. Define CNN model
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 5. Test model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy * 100, "%")

# # 6. Predict one image
# prediction = model.predict(x_test[:1])
# print("Actual digit:", y_test[0])
# print("Predicted digit:", prediction.argmax())

# 7. Display first 10 predictions
plt.figure(figsize=(12, 5))
for i in range(10):
    pred = model.predict(x_test[i:i+1], verbose=0).argmax()
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Actual: {y_test[i]}\nPredicted: {pred}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("mnist_predictions.png")
plt.show()




# import numpy as np
# from torchvision.datasets import MNIST
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input,Dense,Flatten,Conv2D,MaxPool2D
# import matplotlib.pyplot as plt

# train=MNIST(root="dataset\MNIST\data",train=True,download=False)
# test=MNIST(root="dataset\MNIST\data",train=False,download=False)

# x_train,y_train=train.data,np.array(train.targets)
# x_test,y_test=test.data,np.array(test.targets)

# model=Sequential([
#     Input(shape=(28,28,1)),
#     Conv2D(16,(3,3),activation="relu"),
#     MaxPool2D(2,2),
#     Conv2D(32,(3,3),activation="relu"),
#     MaxPool2D(2,2),
#     Flatten(),
#     Dense(128,activation="relu"),
#     Dense(10,activation="softmax")])

# model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=10,batch_size=64)

# for i in range(10):
#     pred=model.predict(x_test[i:i+1],verbose=0).argmax()
#     plt.subplot(2,5,i+1)
#     plt.imshow(x_test[i].squeeze(),cmap="gray")
#     plt.title(f"prediction: {pred}")

# plt.tight_layout()
# plt.show()
