# 匯入機器學習常用模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料集
df = pd.read_csv("train_data_titanic.csv")
# 顯示資料集的前五行
df.head()
df.info()
# 移除Name, Ticket欄位
#df = df.drop(columns=["Name", "Ticket"])

df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
uncommon_titles = ['Rev', 'Dr', 'Col', 'Major', 'Capt', 'Sir', 'Don', 'Countess', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(['Mme', 'Ms', 'Lady'], 'Mrs')
df['Title'] = df['Title'].replace(['Mlle'], 'Miss')
df['Title'] = df['Title'].replace(uncommon_titles, 'Rare')
print(df['Title'].value_counts())
plt.figure(figsize=(10, 6))
sns.barplot(x='Title', y='Survived', data=df)
plt.title('不同稱謂 (Title) 乘客的存活率')
plt.ylabel('平均存活率 (Survived)')
plt.xlabel('乘客稱謂 (Title)')
plt.show()
df = df.drop(columns=["Name"])


# 透過seaborn的pairplot來觀察survived與Fare之間的關係
sns.pairplot(df[['Survived', 'Fare']], dropna=True)
sns.pairplot(df[["Survived", "PassengerId"]], dropna=True)
sns.pairplot(df[["Survived", "Pclass"]], dropna=True)
# 製作一個Sex_num欄位，男生為1，女生為0
df['Sex_num'] = df['Sex'].map({'male': 1, 'female': 0})
sns.pairplot(df[["Survived", "Sex_num"]], dropna=True)

title_dummies = pd.get_dummies(df['Title'], prefix='Title', dtype='int')
if 'Title_Mr' in title_dummies.columns:
    title_dummies = title_dummies.drop(columns=['Title_Mr'])
df = pd.concat([df, title_dummies], axis=1)
df = df.drop(columns=["Title"])

# 刪除 Sex_num 欄位
df = df.drop(columns=["Sex_num"])

df.groupby("Survived").mean(numeric_only=True)

df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()

df.isnull().sum().sort_values(ascending=False)
df.isnull().sum() > len(df)/2 # 判斷哪些欄位缺值超過一半

# 刪除 cabin 欄位
df = df.drop(columns=["Cabin"])

df.groupby('Sex')['Age'].median().plot(kind='bar')

df.groupby('Sex')['Age'].transform('median')

df.fillna({"Age": df.groupby("Sex")["Age"].transform("median")}, inplace=True)

df['Embarked'].value_counts().idxmin()

df.fillna({"Embarked": df['Embarked'].value_counts().idxmax()}, inplace=True)

df = pd.get_dummies(data=df, dtype='int',columns=['Sex', 'Embarked'])
# 刪除 Sex_female欄位
df = df.drop(columns=["Sex_female"])
df = df.drop(columns=["Embarked_Q"])
df = df.drop(columns=["Ticket"])
#df = df.drop(columns=["Sex"])
#df = df.drop(columns=["Embarked"])
df = df.drop(columns=["Pclass"])
df.corr()

X = df.drop(columns=["Survived"])
y = df["Survived"]
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
# 模型評估
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
accuracy_score(y_test, predictions)
recall_score(y_test, predictions)
precision_score(y_test, predictions)
confusion_matrix(y_test, predictions)

pd.DataFrame(
    confusion_matrix(y_test, predictions),
    index=["真的沒存活 0", "真的存活 1"],
    columns=["預測沒存活 0", "預測存活 1"]
)

import joblib
joblib.dump(lr, 'Titanic-LR-20251028.pkl', compress = 3)