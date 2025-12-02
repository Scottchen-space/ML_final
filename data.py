import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv('train.csv')
df.head()
df.info()

df["Extrovert"] = df["Personality"].apply(lambda x: 1 if x == 'Extrovert' else 0)
df["Fear_stage"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
df["Social_drain"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["Personality","Stage_fear","Drained_after_socializing"])

# 資料外向人較多 基本上特徵分布都算廣(沒有哪個特徵特別集中)
# 以下是數值型特徵
sns.pairplot(df[["Extrovert","Time_spent_Alone"]], dropna=True)
sns.pairplot(df[["Extrovert","Social_event_attendance"]], dropna=True)
sns.pairplot(df[["Extrovert","Going_outside"]], dropna=True)
sns.pairplot(df[["Extrovert","Friends_circle_size"]], dropna=True)
sns.pairplot(df[["Extrovert","Post_frequency"]], dropna=True) 

# 以下二元特徵(類別型) 用計數圖
# 繪製二元特徵(fear_stage, social_drain)對應個性的計數圖
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Social_drain', hue='Extrovert')
plt.show()
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Fear_stage', hue='Extrovert')
plt.show() 


# 檢查缺失值
df.isnull().sum().sort_values(ascending=False)
df.isnull().sum() > len(df)/2
df.isnull().sum()

# 計算各特徵值的缺失比例(eg.soial_event_attendance)
total_rows = df.shape[0]
missing_percent = df['Social_event_attendance'].isnull().sum() / total_rows * 100
print(f"缺失比例: {missing_percent:.2f}%")
# 約6~7%


# 填補缺失值
# 數值行用中位數填補
features_to_impute = [
    'Time_spent_Alone', 
    'Social_event_attendance', 
    'Going_outside', 
    'Friends_circle_size', 
    'Post_frequency'
]
imputer = SimpleImputer(strategy='median')
df[features_to_impute] = imputer.fit_transform(df[features_to_impute])

# 類別型用眾數填補
cat_features = ['Fear_stage', 'Social_drain']
mode_imputer = SimpleImputer(strategy='most_frequent')
df[cat_features] = mode_imputer.fit_transform(df[cat_features])


# 檢查二元特徵填補後比例
plt.figure(figsize=(10, 4))
# 畫第一個特徵：Social_drain
plt.subplot(1, 2, 1)
sns.countplot(x=df['Social_drain'])
plt.title('Social_drain After Imputation')

# 畫第二個特徵：Fear_stage
plt.subplot(1, 2, 2)
sns.countplot(x=df['Fear_stage'])
plt.title('Fear_stage After Imputation')
plt.show()

print("Social_drain 比例:\n", df['Social_drain'].value_counts(normalize=True))
print("-" * 20)
print("Fear_stage 比例:\n", df['Fear_stage'].value_counts(normalize=True))

# social drain, fear stage 超過75%的特徵值刪除
df = df.drop(columns=["Social_drain", "Fear_stage"])

df.corr()

X = df.drop(columns=["Extrovert"])
y = df["Extrovert"]
# 將資料集分為訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=67
)

# 建立邏輯迴歸模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
# 訓練模型
lr.fit(X_train, y_train)
# 預測測試集的存活情況
predictions = lr.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
accuracy_score(y_test, predictions)
recall_score(y_test, predictions)
precision_score(y_test, predictions)
confusion_matrix(y_test, predictions)

pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index=["實際內向 (0)", "實際外向 (1)"],
    columns=["預測內向 (0)", "預測外向 (1)"]
)

import joblib
joblib.dump(lr, 'Extrovert_LR_001.pkl', compress = 3)

