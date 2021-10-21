import torch
import torchvision
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST('../data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float() 
y_train = torch.zeros((mnist_train.targets.shape[0], 10)) 
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1 

mnist_test = torchvision.datasets.MNIST('../data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float() 
y_test = torch.zeros((mnist_test.targets.shape[0], 10)) 
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

class SoftmaxModel:
    def __init__(self):
        self.W = torch.ones([784,10], requires_grad=True)
        self.b = torch.ones([1,10], requires_grad=True)
    # Predictor
    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)
    
    def logits(self, x):
        return x @ self.W + self.b
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),
        y.argmax(1)).float())

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x),y)

model = SoftmaxModel()
# Training
optimizer = torch.optim.SGD([model.W, model.b], lr=1)
for epoch in range(1000):
    model.loss(x_train, y_train).backward()  
    optimizer.step() 
    optimizer.zero_grad()  
    if epoch % 100 == 0:
        print("epoch %s, loss %s acc %s " %(epoch, model.loss(x_train, y_train).item(), model.accuracy(x_test, y_test).item()*100))
print("W = %s, b = %s, loss = %s acc = %s" %(model.W, model.b, model.loss(x_train, y_train), model.accuracy(x_test, y_test)))

# Visualization
fig = plt.figure('Photos')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
    plt.title(f'W: {i}')
    plt.xticks([])
    plt.yticks([])

plt.show()