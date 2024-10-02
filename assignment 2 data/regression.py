import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())
print(test_df.head())

# Replace 'Missing' with NaN
train_df.replace('Missing', np.nan, inplace=True)
test_df.replace('Missing', np.nan, inplace=True)

# Convert categorical variables to numerical values using one-hot encoding
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Align the train and test dataframes by columns
train_df, test_df = train_df.align(test_df, join='inner', axis=1)

# Add the target variable back to the train dataframe
train_df['SalePrice'] = pd.read_csv('train.csv')['SalePrice']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
train_df_imputed = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
test_df_imputed = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_df_imputed['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Split data into features and target
X = train_df_imputed.drop(columns=['SalePrice'])
y = train_df_imputed['SalePrice']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
def train_and_evaluate(X_train, y_train, X_val, y_val):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results[name] = rmse
    return results

# Evaluate models
results = train_and_evaluate(X_train, y_train, X_val, y_val)
print("Model evaluation results:", results)

# Cross-validation
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

# Cross-validate models
cv_results = {}
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}
for name, model in models.items():
    cv_results[name] = cross_validate_model(model, X, y)

print("Cross-validation results:")
for name, scores in cv_results.items():
    print(f"{name}: Mean RMSE = {scores.mean()}, Std RMSE = {scores.std()}")

# Train final model
final_model = RandomForestRegressor(random_state=42)
final_model.fit(X, y)

# Visualize feature importances
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Prepare test data
X_test = test_df_imputed.drop(columns=['SalePrice'])

# Predict on test data
test_predictions = final_model.predict(X_test)

# Save predictions
output = pd.DataFrame({'ID': pd.read_csv('test.csv')['ID'], 'SalePrice': test_predictions})
output.to_csv('predictions.csv', index=False)