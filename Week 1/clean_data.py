import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Week 1\train.csv")
# print(data.shape)
# print(data.head())

# y_train = data['label']
# x_train = data.drop('label',axis=1)

# image = x_train.iloc[8].values
# image = image.reshape(28,28)

for x in data :
    print(x)