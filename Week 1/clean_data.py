import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"Week 1\train.csv")

data_re =  data.replace('',np.nan)
missing_values = data_re.isnull().sum()
print(missing_values)

print('-' * 200)

morethan_255 = data_re > 255
print(morethan_255.sum())

y_train = data_re["label"]
x_train = data.drop('label', axis=1)

for x in range (10):
    if x in y_train.values:
        plt.subplot(5, 2, x + 1)
        plt.imshow(x_train[y_train == x].values[0].reshape(28, 28), cmap='gray')
plt.show()