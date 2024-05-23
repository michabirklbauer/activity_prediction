import numpy as np
import matplotlib.pyplot as plt

# Define a sample linear function
def f(x, theta):
  return theta[0] * x + theta[1]

# Define the cost function (mean squared error)
def cost_function(x, y, theta):
  m = len(x)
  predictions = f(x, theta)
  return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Define the learning rate
alpha = 0.01

# Generate some sample data
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 5, 4])

# Initialize theta
theta = np.array([0, 0])

# Lists to store theta values and cost function values during each iteration
theta_history = [theta.copy()]
cost_history = [cost_function(x, y, theta)]

# Perform gradient descent for a certain number of iterations
iterations = 1000
for i in range(iterations):
  # Calculate the gradient of the cost function
  gradient = (1 / len(x)) * np.dot(x, (f(x, theta) - y))

  # Update theta using the gradient descent rule
  theta -= int(alpha * gradient)

  # Store theta and cost for plotting
  theta_history.append(theta.copy())
  cost_history.append(cost_function(x, y, theta))

# Plot the data points
plt.scatter(x, y, label='Data Points')

# Plot the fitted line using the final theta
x_fit = np.linspace(min(x), max(x), 100)
y_fit = f(x_fit, theta)
plt.plot(x_fit, y_fit, label='Fitted Line')

# Plot the path of theta over iterations (optional)
theta0_history = [t[0] for t in theta_history]
theta1_history = [t[1] for t in theta_history]
plt.plot(theta0_history, theta1_history, label='Theta Trajectory')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent for Linear Regression')
plt.legend()

plt.show()
