import numpy as np

# Training data
x_train = np.array([1.0, 2.0])       # House sizes in 1000 sqft
y_train = np.array([300.0, 500.0])   # House prices in thousand dollars

# Compute cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = cost / (2 * m)
    return total_cost

# Compute gradients
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0.0
    dj_db = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# Gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

# Initialize parameters
w_init = 0.0
b_init = 0.0
iterations = 10000
alpha = 0.01

# Train the model
w_final, b_final = gradient_descent(x_train, y_train,
                                    w_init, b_init,
                                    alpha, iterations,
                                    compute_cost, compute_gradient)

# Take user input for house size and predict price
house_size = float(input("Enter house size in 1000 sqft: "))
predicted_price = w_final * house_size + b_final
print(f"Predicted house price: {predicted_price:0.1f} Thousand dollars")
