import numpy as np
import matplotlib.pyplot as plt

# dataset
x_train = np.array([1.0 , 2.0])
y_train = np.array([300.0 , 500.0])

# Linear model
def compute_model_output(x , w , b):
    return w*x + b

# Parameters
w = 200
b = 100
tmp_f_wb = compute_model_output(x_train , w , b)

# plot
plt.plot(x_train , tmp_f_wb , c = 'b' , label = 'predicted value')
plt.scatter(x_train , y_train , c = 'r' , marker = 'x' , label = 'actual value' )
plt.title("Housing Prices")
plt.xlabel('size (1000 sqft)')
plt.ylabel('price (1000 $)')
plt.legend()
plt.show(block=False)

# predict price for user input
try:
    house_size = float(input("Enter house size in sqft: "))  
    x_new = house_size / 1000.0  
    predicted_price = compute_model_output(np.array([x_new]), w, b)[0]
    print(f"Predicted price for {house_size:.0f} sqft: ${predicted_price:.0f}k")
except ValueError:
    print("Please enter a valid number for house size!")

