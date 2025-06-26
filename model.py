import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib

df = pd.read_csv('Concrete.csv')
x = df.iloc[:,:-1]# every column except last one
y = df.iloc[:,-1] # last one
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
model = RandomForestRegressor(n_estimators =100,random_state = 42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

#User Inputs
cement = float(input("Enter Cement(kg/m3): "))
blast_furnance = float(input("Enter Blast Furnace slag(kg/m3): "))
fly_ash = float(input("Enter Fly Ash(kg/m3): "))
water = float(input("Enter Water(kg/m3): "))
super = float(input("Enter Superplasticier(kg/m3): "))
coarse = float(input("Enter Coarse Aggregate(kg/m3): "))
fine = float(input("Enter Fine Aggregate(kg/m3): "))
age = float(input("Enter Age(days): "))
final = np.array([[cement,blast_furnance,fly_ash,water,super,coarse,fine,age]])
out = model.predict(final)
print("The Concrete Compressive Strength(Mpa) is : {:.2f}".format(out[0]))

# Save trained model
joblib.dump(model, 'concrete_strength_model.pkl')
print("Model saved successfully!")