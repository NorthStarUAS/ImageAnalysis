#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import square

from scipy.integrate import quad

from math import* # import all function from math

# x axis has been chosen from –π to +π, value of 1 smallest square
# along x axis is 0.001
x = np.arange(-4*np.pi,4*np.pi,0.001)

# defining square wave function 𝑦 =−1, 𝑓𝑜𝑟 − 𝜋 ≤ 𝑥 ≤ 0 y= +1, 𝑓𝑜𝑟 0 ≤ 𝑥 ≤ 𝜋
y = square(x)

# define fuction
fc = lambda x:square(x)*cos(i*x)  # i :dummy index
fs = lambda x:square(x)*sin(i*x)

n = 2   # max value of I, not taken infinity, better result with high value
An = [] #  defining array
Bn = []
sum = 0

for i in range(n):
    an = quad(fc,-np.pi,np.pi)[0]*(1.0/np.pi)
    An.append(an)

for i in range(n):
    bn = quad(fs,-np.pi,np.pi)[0]*(1.0/np.pi)
    Bn.append(bn) # putting value in array Bn

for i in range(n):
    if i == 0:
        sum = sum + An[i]/2
    else:
        sum = sum + (An[i]*np.cos(i*x) + Bn[i]*np.sin(i*x))

plt.figure()
plt.plot(x,sum,'g')
plt.plot(x,y,'r--')
plt.title("fourier series for square wave, n=%d" % n)

plt.figure()
plt.plot(An, label="A[n]")
plt.plot(Bn, label="B[n]")
plt.legend()
plt.show()

