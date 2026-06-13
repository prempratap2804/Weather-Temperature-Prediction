import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split





df= pd.read_csv("DATASET/weather_prediction_dataset.csv")
#print(df.head())

x = df.iloc[:, 1:]   # features
y = df.iloc[:, 0]    # target = temperature

#print(x.head(5))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)

print(lr.coef_)
print(lr.intercept_)

new_data = pd.DataFrame({
    "humidity_percent": [70],
    "wind_speed_kmh": [15],
    "cloud_cover_percent": [50],
    "pressure_hpa": [1012],
    "dew_point_c": [20],
    "rainfall_mm": [2.5]
})

prediction = lr.predict(new_data)

print("Predicted Temperature:", prediction[0])

# Test data predictions
y_pred = lr.predict(x_test)
'''
# Graph
plt.bar(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()
'''

plt.scatter(y_test, y_pred, label="Test Data")

# New prediction point
plt.scatter([prediction[0]], [prediction[0]], marker="x", s=100, label="New Prediction")

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.legend()
plt.show()