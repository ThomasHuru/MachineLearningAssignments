import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

# Observed/training input and output
file_out = pd.read_csv('day_head_circumference.csv')
x_init = file_out.iloc[1:1001,0].values
y_init = file_out.iloc[1:1001,1].values
x_train = torch.tensor(x_init, dtype=torch.float32).reshape(-1,1)
y_train = torch.tensor(y_init, dtype=torch.float32).reshape(-1,1)



class NonLinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return (20*torch.sigmoid((x @ self.W + self.b))+31)  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(10000):      
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.linspace(torch.min(x_train), torch.max(x_train), 10).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()