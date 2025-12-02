# import model
import joblib
model_pretrained = joblib.load("Extrovert_LR_001.pkl")

import pandas as pd
from sklearn.impute import SimpleImputer
df_test = pd.read_csv("test.csv")
df_test.info()
df_test = df_test.drop(columns=["Stage_fear"])
df_test = df_test.drop(columns=["Drained_after_socializing"])

features_to_impute = [
    'Time_spent_Alone', 
    'Social_event_attendance', 
    'Going_outside', 
    'Friends_circle_size', 
    'Post_frequency'
]
# 2. 建立填補器 (設定策略為中位數 'median')
imputer = SimpleImputer(strategy='median')
# 3. 進行填補
df_test[features_to_impute] = imputer.fit_transform(df_test[features_to_impute])



# predict
predictions2 = model_pretrained.predict(df_test)

Personality = ['Extrovert' if pred == 1 else 'Introvert' for pred in predictions2]
# save to csv
forSubmissionDF = pd.DataFrame(
    {"id": df_test["id"], 
     "Personality": Personality})
forSubmissionDF.to_csv("for_submission_001.csv", index=False)