import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('Delhi_v2.csv')
data.drop(columns=['Landmarks','Lift','Furnished_status','Unnamed: 0','latitude','longitude','Status','desc','Balcony','parking','neworold','type_of_building','Price_sqft',],inplace=True)
data['Address'] = data['Address'].apply(lambda x: x.strip())
Address_counts = data['Address'].value_counts()
Address_count_less_10 = Address_counts[Address_counts<=10]
Address_count_less_10
data['Address']=data['Address'].apply(lambda x: 'other' if x in Address_count_less_10 else x)
data.to_csv("Cleaned_data.csv")
X=data.drop(columns=['price'])
y=data['price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
column_trans = make_column_transformer((OneHotEncoder(sparse_output=False),['Address']), remainder = 'passthrough')
scaler = StandardScaler(with_mean=False)
lr = LinearRegression()
pipe = make_pipeline(column_trans, scaler, lr)
pipe.fit(X_train,y_train)
y_pred_lr = pipe.predict(X_test)
r2_score(y_test, y_pred_lr)
lasso = Lasso(max_iter=100000)
pipe = make_pipeline(column_trans,scaler, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso = pipe.predict(X_test)
r2_score(y_test, y_pred_lasso)
ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(X_train, y_train)
y_pred_ridge = pipe.predict(X_test)
r2_score(y_test, y_pred_ridge)
import pickle
pickle.dump(pipe, open('RidgeModel.pkl','wb'))