import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Bookings.csv')
#print(data)
#print(data.isnull().sum())
#print(data.describe())  
#print(data.info())

# Data Cleaning

#Fill null values with unknown in Cancled Rides By Customer
data['Canceled_Rides_by_Customer'].fillna('Unknown', inplace=True)
#Fill null values with unknown in Cancled Rides By Driver
data['Canceled_Rides_by_Driver'].fillna('Unknown', inplace=True)
#Fill Incomplete Rides with 0
data['Incomplete_Rides_Reason'].fillna(0, inplace=True)
#drop null values in Payment Method
data.dropna(subset=['Payment_Method'], inplace=True)
# Fill Customer Rating
data['Customer_Rating'].fillna(data['Customer_Rating'].mean(), inplace=True)
# Fill Driver Rating
data['Driver_Ratings'].fillna(data['Driver_Ratings'].mean(), inplace=True)
print(data)
#drop Unnamed: 17
data.drop(columns=['Unnamed: 17'], inplace=True)
#print(data.isnull().sum())

# Exploratory Data Analysis

# Countplot of Incomplete Rides
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Incomplete_Rides_Reason', palette='viridis')
plt.title('Incomplete Rides')
plt.xlabel('Incomplete Rides Reason')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Countplot of Payment Method
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Payment_Method', palette='viridis')
plt.title('Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Countplot of Customer Rating
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Customer_Rating', palette='viridis')
plt.title('Customer Rating')
plt.xlabel('Customer Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='Pastel1', fmt='.2f')
#sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#Linear Regression

# Define features and target

X = data[['Ride_Distance','Customer_Rating']]
y = data['Booking_Value']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Coefficients:", dict(zip(X, model.coef_)))
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

#Export cleaned csv file
#data.to_csv('Bookings processed.csv', index=False)