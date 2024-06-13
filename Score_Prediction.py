import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

data=pd.DataFrame({
    "Hours Studied":[2,3.5,5,6.5,8,9.5],
    'exam Score':[50,60,75,80,90,95]
})
X=data[['Hours Studied']]
y=data['exam Score']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f' MAE:{mae}, MSE:{mse}, R-squared:{r2}')
new_data=[[7]]
predicted_score = model.predict(new_data)
print(f' Predicted score ={predicted_score[0]}')

plt.scatter(X,y,color="blue", label='Data')
plt.plot(X,model.predict(X),color='red', label="Regression Line")

plt.xlabel("No.of Hours studied")
plt.ylabel('Exam Score')
plt.title('Graph')
plt.show()
