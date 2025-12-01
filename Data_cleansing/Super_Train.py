import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ โหลด Data
# ==============================
data = pd.read_csv(r"Data/train.csv")
data_original = data.copy()  # เก็บต้นฉบับไว้เปรียบเทียบ

# ==============================
# 2️⃣ แปลง empty string / whitespace เป็น NaN
# ==============================
empty_string_mask = data.applymap(lambda x: str(x).strip() == '')
empty_rows, empty_cols = np.where(empty_string_mask)
for r, c in zip(empty_rows, empty_cols):
    print(f"🔹 Empty string → แปลงเป็น NaN ที่ Row {r}, Column '{data.columns[c]}'")
    data.at[r, data.columns[c]] = np.nan

# ==============================
# 3️⃣ แปลงค่าที่ควรเป็น numeric แต่มี string ผิดรูป
# ==============================
object_cols = data.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    for r in range(len(data)):
        val = data.at[r, col]
        if pd.notnull(val):
            try:
                float(val)
            except:
                # แสดงเฉพาะค่า non-numeric จริง ๆ
                print(f"🔹 Non-numeric → แปลงเป็น NaN ที่ Row {r}, Column '{col}' (ค่าเดิม: {val})")
                data.at[r, col] = np.nan
    # แปลง column เป็น numeric (float) หลังจากแก้ค่า non-numeric
    if data[col].isna().sum() > 0:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ==============================
# 4️⃣ เติมค่า NaN แทนการลบ
# ==============================
numeric_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()
for col in numeric_cols:
    nan_rows = np.where(data[col].isna())[0]
    if len(nan_rows) > 0:
        mean_val = data[col].mean()
        for r in nan_rows:
            print(f"🔹 เติมค่า NaN ด้วย mean ({mean_val}) ที่ Row {r}, Column '{col}'")
            data.at[r, col] = mean_val

for col in object_cols:
    if col in data.columns:
        nan_rows = np.where(data[col].isna())[0]
        if len(nan_rows) > 0:
            mode_val = data[col].mode()[0]
            for r in nan_rows:
                print(f"🔹 เติมค่า NaN ด้วย mode ({mode_val}) ที่ Row {r}, Column '{col}'")
                data.at[r, col] = mode_val

# ==============================
# 5️⃣ ค่าเกิน 255 → clip
# ==============================
for col in numeric_cols:
    over_255_rows = np.where(data[col] > 255)[0]
    for r in over_255_rows:
        print(f"🔹 ค่าเกิน 255 → clip ที่ Row {r}, Column '{col}' (ค่าเดิม: {data.at[r, col]})")
        data.at[r, col] = 255

# ==============================
# 6️⃣ จัดการ categorical ให้ consistent (เฉพาะ categorical จริง ๆ)
# ==============================
for col in object_cols:
    # ข้าม column ที่ numeric ได้เลย
    if col in numeric_cols:
        continue
    for r in range(len(data)):
        val_old = data.at[r, col]
        val_new = str(val_old).lower().strip()
        if val_old != val_new:
            print(f"🔹 ทำให้ consistent → แปลงเป็น '{val_new}' ที่ Row {r}, Column '{col}' (ค่าเดิม: '{val_old}')")
            data.at[r, col] = val_new

# ==============================
# 7️⃣ Visualize ตัวอย่าง (dataset image 28x28)
# ==============================
if 'label' in data.columns:
    X_vis = data.drop('label', axis=1).astype(float)
    y_vis = data['label']
    for x_val in range(10):
        if x_val in y_vis.values:
            plt.subplot(5, 2, x_val + 1)
            plt.imshow(X_vis[y_vis == x_val].values[0].reshape(28,28), cmap='gray')
    plt.show()

# ==============================
# 8️⃣ เซฟ Data ที่ clean แล้ว
# ==============================
data.to_csv("Data/train_cleaned.csv", index=False)
print("✅ Data Cleaning เสร็จแล้ว และเซฟเป็น train_cleaned_full.csv")
