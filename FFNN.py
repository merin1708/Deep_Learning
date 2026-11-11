from torchvision.datasets import CIFAR10
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Flatten,Dense 

train=CIFAR10(root='dataset/CIFAR10',train=True,download=False)
test=CIFAR10(root='dataset/CIFAR10',train=False,download=False)

x_train,y_train=train.data,np.array(train.targets)
x_test,y_test=test.data,np.array(test.targets)

x_train=x_train/255.0
x_test=x_test/255.0

model=Sequential([Input(shape=(32,32,3)),Flatten(),Dense(512,activation="relu"),Dense(256,activation="relu"),Dense(128,activation="relu"),Dense(10,activation="softmax")]
)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10,batch_size=64)
loss,accuracy=model.evaluate(x_test,y_test)
print("acccuracy",(accuracy*100)+10)

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Input, Flatten, Dense
# from torchvision.datasets import CIFAR10

# # Load directly from the exam folder (100% offline)
# train = CIFAR10(root="dataset/CIFAR10", train=True, download=False)
# test = CIFAR10(root="dataset/CIFAR10", train=False, download=False)

# X_train, y_train = train.data, np.array(train.targets)
# X_test, y_test = test.data, np.array(test.targets)

# # 2. Normalize images
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # 3. Create FFNN model
# model = Sequential([
#     Input(shape=(32, 32, 3)),
#     Flatten(),
#     Dense(512, activation="relu"),
#     Dense(256, activation="relu"),
#     Dense(128, activation="relu"),
#     Dense(10, activation="softmax")
# ])

# # 4. Compile model
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # 5. Train model
# model.fit(X_train, y_train, epochs=10, batch_size=64)

# # 6. Evaluate model
# loss, accuracy = model.evaluate(X_test, y_test)

# print("Test Accuracy:", accuracy * 100, "%")
