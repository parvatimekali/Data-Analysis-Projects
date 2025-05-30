import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('BSE_SENSEX.csv')
print(df)
print(df.describe())
print(df.info())
print(df.isnull().sum())

#Convert the 'Date' column to datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'],dayfirst=True)
df['Date'] = df['Datetime'].dt.date
df['Time'] = df['Datetime'].dt.time

#EDA
plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
sns.lineplot(x='Datetime',y='Close',data=df)
plt.title('Close Price over Time')

#Volume over time
plt.subplot(2,2,2)
sns.lineplot(x='Datetime',y='Volume',data=df)
plt.title('Volume over Time')
#plt.show()

#Correlation heatmap
plt.subplot(2,2,3)
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm')
#corr = df.select_dtypes(include=['float64', 'int64']).corr()
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
#plt.show()

#Distribution of Trades
plt.subplot(2,2,4)
sns.histplot(df['Trades'], bins=50, kde=True)
plt.title('Distribution of Trades')
plt.tight_layout()
plt.show()

#Linear regression Code
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

#Split into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a linear regression model
lr = LinearRegression()
#Fit the model
lr.fit(X_train, y_train)

#predictions and evalution
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

#Plotting the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Price')
plt.show()

#Export cleaned csv file
#df.to_csv('BSE_SENSEX processed.csv',index=False)