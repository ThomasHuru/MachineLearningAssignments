import torch
import matplotlib.pyplot as plt
import numpy as np



x_train = torch.tensor([[1.0],[0.0]]).reshape(-1,1)
y_train = torch.tensor([[0.0],[1.0]]).reshape(-1,1)

class NOTOperatorModel:
    def __init__(self):
        self.W = torch.rand((1,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)
    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x),y)

model = NOTOperatorModel()

optimizer = torch.optim.SGD([model.b, model.W], lr=0.1)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  
    optimizer.step() 
    optimizer.zero_grad()  

print("W = %s, b = %s, loss = %s" %(model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0,1.0, 0.01).reshape(-1,1)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = \sigma(xW + b)$')
plt.legend()
plt.show()