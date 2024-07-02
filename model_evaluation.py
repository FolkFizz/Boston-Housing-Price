import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
train_data = pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\Data Science Project Practice\Housing_Price\train.csv")
test_data = pd.read_csv(r"C:\Users\Acer\OneDrive\Desktop\Data Science Project Practice\Housing_Price\test.csv")

# Display the first few rows of the training data
print("First few rows of the training data:")
print(train_data.head())

# Get information about the dataset
print("\nDataset Information:")
print(train_data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(train_data.describe())

# Check for missing values
print("\nMissing Values:")
print(train_data.isnull().sum())

# Visualize the distribution of the target variable (medv)
plt.figure(figsize=(10, 6))
sns.histplot(train_data['medv'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Median Value (in $1000s)')
plt.savefig('house_price_distribution.png')
plt.close()

# Correlation matrix
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_matrix.png')
plt.close()

# Prepare the data for modeling
X = train_data.drop(['ID', 'medv'], axis=1)
y = train_data['medv']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print("\nModel Evaluation:")
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared Score: {r2}')

# Make predictions on the test set
X_test = test_data.drop('ID', axis=1)
test_predictions = model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'medv': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

# Save the model
joblib.dump(model, 'linear_regression_model.joblib')
print("Model saved as: linear_regression_model.joblib")

# สร้างและบันทึก feature_importance.csv
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to: feature_importance.csv")

# สร้างและบันทึก model_metrics.joblib
metrics = {
    'mse': mse,
    'rmse': rmse,
    'r2': r2
}
joblib.dump(metrics, 'model_metrics.joblib')
print("Model metrics saved to: model_metrics.joblib")

try:
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance plot saved as: feature_importance.png")
except Exception as e:
    print(f"An error occurred while plotting feature importance: {e}")