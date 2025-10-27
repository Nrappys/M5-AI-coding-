import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"Week 1\train.csv") 

data_re =  data.replace('',np.nan) #เปลี่ยนempty เป็น NaN
missing_values = data_re.isnull().sum() #นับ NaN หรือ null ในแต่ละ column มาบวกกัน
print(missing_values)

data_re = data_re.dropna() # ลบ missing values ทั้งหมด

print('-' * 200) # กันตาระเบิด

morethan_255 = data_re > 255  #หาcellที่มีค่าเกิน 255
print(morethan_255.sum()) 

y_train = data_re["label"] # เลือก column label 
x_train = data.drop('label', axis=1) #cell ทั้งหมดใน column ยกเว้น label

for x in range (10):
    if x in y_train.values: #หาว่าเลข x อยู่ใน column label นั้นมั้ย
        plt.subplot(5, 2, x + 1) # subplot 5x2
        plt.imshow(x_train[y_train == x].values[0].reshape(28, 28), cmap='gray') # x_train[y_train == x] เอาแถวที่ label = x  
plt.show()                                                                       # .values[0] เลือกแค่แถวแรก
                                                                                 # .reshape(28, 28) เปลี่ยนรูปเป็น 28x28 = 784 pixels