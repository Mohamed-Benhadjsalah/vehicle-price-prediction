#Import our packages and librairies
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import xgboost as xgb 
## Download data from the csv file ##
df =pd.read_csv('/Users/donaldtrump/desktop/Projects/vehicules/vehicles.csv',nrows=5000)
print("Data loaded successfully!")
print(df.head())
df.shape
df.info()
df.describe()
Data_Cleaning

df.drop(['id','url','region','region_url','image_url','county','state','manufacturer','posting_date'],axis=1,inplace=True)
print(df.columns.tolist())
print(df.isnull().sum())
df.drop(columns=['size','condition','VIN','description'],inplace=True)
for j in ['year','odometer','lat','long']:
    df[j]=df[j].fillna(df[j].mean())
cat_col=['model','cylinders','fuel','title_status','transmission','drive','type','paint_color']
for i in cat_col:
    df[i]=df[i].fillna(df[i].mode()[0])
print(df.columns.tolist())    
print("any missing values left ",df.isnull().values.any())

##  Explore Data ##
print(df.info())
print(df.describe())
#Check outliers
plt.figure(figsize=(10,6))
sns.histplot(df['price'],bins=50,kde=True)
plt.title('Distributin of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

sns.scatterplot(data=df ,x='year',y='price')
plt.title('Price vs year')
plt.show()

## Linear regression model ##
df1 = pd.get_dummies(df, columns=cat_col, drop_first=True)
df2= df1[(df1['price'] > 500) & (df1['price'] < 100000)]
lr=LinearRegression()
X=df2.drop('price',axis=1)
y=df2['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(r2)
correlations = df.corr(numeric_only=True)['price'].drop('price').sort_values()
# Plot show weak correlation
plt.figure(figsize=(8, 5))
sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
plt.title('Correlation of Numeric Features with Price')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

## Random forest Regressor ##
RF=RandomForestRegressor(random_state=42)
RF.fit(X_train,y_train)
y_pred1=RF.predict(X_test)
mse1=mean_squared_error(y_test,y_pred1)
r3=r2_score(y_test,y_pred1)
print("The Random forest MSE:",mse1,"and R2 score is:"r3)
sns.histplot(df2['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.show()
#Important features
importances = RF.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp[:15], y=feat_imp.index[:15])
plt.title("Top 15 Feature Importances")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df2, x='odometer', y='price', alpha=0.5)
plt.title('Price vs Odometer')
plt.xlabel('Odometer (miles)')
plt.ylabel('Price ($)')
plt.show()

## XGBoost model ##
XB= xgb.XGBRegressor(n_estimators=100,random_state=42)
XB.fit(X_train,y_train)
y_pred2=XB.predict(X_test)
mse2=mean_squared_error(y_test,y_pred2)
r4=r2_score(y_test,y_pred2)
print("the XGBOOST mean squred error is :",mse2,"and R2 score is:",r4)