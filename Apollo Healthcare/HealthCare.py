import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Apollo-Healthcare.csv')
print(data)
#print(data.isnull().sum())
#print(data.describe())
#print(data.info())

# Exploratory Data Analysis (EDA)

# Distribution of numerical columns
data.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

# Value counts for categorical features
for col in data.select_dtypes(include='object').columns:
    print(f"\nValue counts for {col}:\n", data[col].value_counts())

# Example: Top diagnoses or departments (if such columns exist)
if 'Test' in data.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, y='Test', order=data['Test'].value_counts().index)
    plt.title('Patients by Test')
    plt.xlabel('Number of Patients')
    plt.show()

# Linear Regression 

# Define features and target
target = 'Billing Amount'
features = ['Feedback', 'Health Insurance Amount']
X = data[features]
y = data[target]

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Visualization: Actual vs Predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Billing Amount")
plt.ylabel("Predicted Billing Amount")
plt.title("Actual vs Predicted Billing Amount")
plt.grid(True)
plt.show()

#Export cleaned csv data
#data.to_csv('Apollo-Healthcare-processed.csv', index=False)