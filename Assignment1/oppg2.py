import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

# Observed/training input and output
file_out = pd.read_csv('day_length_weight.csv')
x_init = file_out.iloc[1:1001,0:2].values
y_init = file_out.iloc[1:1001,2].values
x_train = torch.tensor(x_init, dtype=torch.float32).reshape(-1,2)
y_train = torch.tensor(y_init, dtype=torch.float32).reshape(-1,1)
print(x_train[1])




class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0],[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(500000):      
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot(x_train[:, 0].reshape(-1), x_train[:, 1].reshape(-1), y_train.reshape(-1), 'o', label='$(x_1^{(i)}, x_2^{(i)}, y^{(i)})$')
ax.set_xlabel('$x_1$ (length)')
ax.set_ylabel('$x_2$ (weight)')
ax.set_zlabel('y (day)')

x_grid, y_grid = torch.meshgrid(torch.linspace(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 10),
                                torch.linspace(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 10))
z_grid = torch.empty(x_grid.shape)
for i in range(0, z_grid.shape[0]):
    for j in range(0, z_grid.shape[1]):
        z_grid[i, j] = model.f(torch.tensor([[x_grid[i, j], y_grid[i, j]]]))
ax.plot_wireframe(x_grid, y_grid, z_grid.detach(), color='green', label='$\\hat y = f(\\mathbf{x}) = \\mathbf{xW}+b$')

ax.legend()
plt.show()
