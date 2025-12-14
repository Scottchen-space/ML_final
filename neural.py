import pandas as pd

df = pd.read_csv("train.csv")

df.head()
df.shape

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Input, Dense
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 顯示訓練和驗證損失的圖表
import matplotlib.pyplot as plt

def draw_loss(history):
    loss = history.history["loss"]
    epochs = range(1, len(loss) + 1)
    if "val_loss" in history.history:
        val_loss = history.history["val_loss"]
        plt.plot(epochs, val_loss, "r", label="Validation Loss")
        plt.title("Training and Validation Loss")
    else:
        plt.title("Training Loss")
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def draw_acc(history):
    acc = history.history["accuracy"]
    epochs = range(1, len(acc) + 1)
    if "val_accuracy" in history.history:
        # 如果有驗證準確度，則繪製驗證準確度
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, val_acc, "r--", label="Validation Acc")
        plt.title("Training and Validation Accuracy")
    else:
        # 否則只繪製訓練準確度
        plt.title("Training Accuracy")

    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# 轉換類別資料
df["Extrovert"] = df["Personality"].apply(lambda x: 1 if x == 'Extrovert' else 0)
df["Fear_stage"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
df["Social_drain"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["Personality","Stage_fear","Drained_after_socializing", "id"])

# 填補缺失值
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


# np.random.seed(10)  # 指定亂數種子
# 載入糖尿病資料集
# dataset = df.values
# np.random.shuffle(dataset)  # 使用亂數打亂資料
# 分割成特徵資料和標籤資料
Y = df["Extrovert"].values
X = df.drop(columns=["Extrovert"]).values
print("特徵數量:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# 定義模型
model = Sequential()
model.add(Input(shape=(7,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()  # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model.compile(loss="binary_crossentropy", 
              optimizer="sgd", metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=30, batch_size=10)
print("--------------------------")
# 評估模型
loss, accuracy = model.evaluate(X_train, y_train)
print("training data 準確度 = {:.6f}".format(accuracy))  # 0.969026
loss, accuracy = model.evaluate(X_test, y_test)
print("test data 準確度 = {:.6f}".format(accuracy)) # 0.966802

draw_loss(history)  
draw_acc(history) 

# 特徵標準化
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
# 定義模型
model2 = Sequential()
model2.add(Input(shape=(7,)))
model2.add(Dense(10, activation="relu"))
model2.add(Dense(8, activation="relu"))
model2.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
print("--------------------------")
# 編譯模型
model2.compile(loss="binary_crossentropy", 
               optimizer="sgd", metrics=["accuracy"])
# 訓練模型
history2 = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)
print("--------------------------")
# 評估模型
loss, accuracy = model2.evaluate(X_train, y_train)
print("training data 準確度 = {:.6f}".format(accuracy))
loss, accuracy = model2.evaluate(X_test, y_test)
print("test data 準確度 = {:.6f}".format(accuracy))

draw_loss(history2)
draw_acc(history2)

# 優化器
model3 = Sequential()
model3.add(Input(shape=(7,)))
model3.add(Dense(10, activation="relu"))
model3.add(Dense(8, activation="relu"))
model3.add(Dense(1, activation="sigmoid"))
model.summary()   # 顯示模型摘要資訊
# 編譯模型
print("--------------------------")
model3.compile(loss="binary_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history3 = model3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=10)
print("--------------------------")
# 評估模型
loss, accuracy = model3.evaluate(X_train, y_train)
print("training data 準確度 = {:.6f}".format(accuracy))
loss, accuracy = model3.evaluate(X_test, y_test)
print("test data 準確度 = {:.6f}".format(accuracy))

draw_loss(history3)
draw_acc(history3)


model3_2 = Sequential()
model3_2.add(Input(shape=(7,)))
model3_2.add(Dense(10, activation="relu"))
model3_2.add(Dense(8, activation="relu"))
model3_2.add(Dense(1, activation="sigmoid"))
model3_2.summary()   # 顯示模型摘要資訊
# 編譯模型
print("--------------------------")
model3_2.compile(loss="binary_crossentropy", 
               optimizer="adam", metrics=["accuracy"])
# 訓練模型
history3_2 = model3_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=10)
print("--------------------------")
# 評估模型
loss, accuracy = model3.evaluate(X_train, y_train)
print("training data 準確度 = {:.6f}".format(accuracy))
loss, accuracy = model3.evaluate(X_test, y_test)
print("test data 準確度 = {:.6f}".format(accuracy))

draw_loss(history3_2)
draw_acc(history3_2)

