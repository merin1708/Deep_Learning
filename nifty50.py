# from torch.optim import optimizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Input
import matplotlib.pyplot as plt

data=pd.read_csv("nifty50.csv",header=[0,1],index_col=0)
prices=data[['Close']].dropna().values

split=int(len(prices)*0.8)
scalar=MinMaxScaler()
scalar.fit(prices[:split])
prices=scalar.transform(prices)

x=[]
y=[]

for i in range(60,len(prices)):
    x.append(prices[i-60:i])
    y.append(prices[i])

x=np.array(x)
y=np.array(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)

model=Sequential([Input(shape=(60,1)),
       LSTM(64,return_sequences=True),
       LSTM(64),
       Dense(1)])

model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=10, batch_size=64)
prediction=model.predict(x_test)

actual=scalar.inverse_transform(y_test)
predicted=scalar.inverse_transform(prediction)

plt.figure(figsize=(12,5))
plt.plot(actual)
plt.plot(predicted)
plt.show()


