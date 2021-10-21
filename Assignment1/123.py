import math
import numpy
print("chpt0.1 cpi:")
x=0
for i in range (0,51):
    x=x+1.0000**i
print(x)
x=1.0001
print((x**51-1)/(x-1))