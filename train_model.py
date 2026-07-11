import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

csv_path = 'Housing (2).csv'
model_path = 'house_price_model.pkl'
encoder_path = 'lable_encoders.pkl'

if not os.path.exists(csv_path):
    raise FileNotFoundError(csv_path)

df = pd.read_csv(csv_path)

categorical_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
features = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']

encoders = {}
encoded_df = df.copy()
for col in categorical_cols:
    encoder = LabelEncoder()
    encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))
    encoders[col] = encoder

X = encoded_df[features]
y = encoded_df['price']
model = LinearRegression()
model.fit(X, y)

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
with open(encoder_path, 'wb') as f:
    pickle.dump(encoders, f)

print('MODEL_SAVED', model_path)
print('ENCODERS_SAVED', encoder_path)
print('R2_SCORE', model.score(X, y))
