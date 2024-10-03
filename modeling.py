import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

input_file = 'cleaned_indiana_top_10.csv'
df = pd.read_csv(input_file)

df['Fwd P/E'] = df['Fwd P/E'].str.replace('x', '', regex=False).astype(float)
df['P/Norm EPS'] = df['P/Norm EPS'].str.replace('x', '', regex=False).astype(float)
df['TEV/Tot Rev'] = df['TEV/Tot Rev'].str.replace('x', '', regex=False).astype(float)

df['Future Tot Rev, 1Y Gr %'] = df['Tot Rev, 1Y Gr %'].shift(-1)
df = df.dropna(subset=['Future Tot Rev, 1Y Gr %']) 

#target variables and features
target = df['Future Tot Rev, 1Y Gr %']
features = df.drop(columns = ['Tot Rev, 1Y Gr %', 'Future Tot Rev, 1Y Gr %', 'Company', 'Overview', 'Ticker'])

features = features.dropna()
target = target[features.index]

#train test and split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#train the model (linear reg)
model = LinearRegression()
model.fit(x_train, y_train)

#make predictions
y_pred = model.predict(x_test)

#mse and r2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2-score: {r2}')

predictions_output_csv = 'model_predictions_future_growth.csv'
predictions_df = pd.DataFrame({'Actual future Growth': y_test, 'Predicted future Growth': y_pred})
predictions_df.to_csv(predictions_output_csv, index=False)
print(f'Model predictions saved to  {predictions_output_csv}')