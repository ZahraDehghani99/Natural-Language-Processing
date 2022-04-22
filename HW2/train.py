import numpy as np
from hmm import *
 
initial_pro, A, B = [], [], []

with open('hmm_data/model_init.txt', "r") as f:
    mylist = [line.rstrip('\n') for line in f]
    for i, line in enumerate(mylist):
        if i in range(1, 2):
            initial_pro.append([float(y) for y in line.split()])
        if i in range(4, 10):  
            A.append([float(y) for y in line.split()])
        if i in range(12, 18):    
            B.append([float(y) for y in line.split()])

initial_pro = np.asarray(initial_pro)
A = np.asarray(A)
B = np.asarray(B)
# initial_pro = np.array([[1, 0]])
# A = np.array([[0.6, 0.4], [0, 1]])
# B = np.array([[0.8, 0.3], [0.2, 0.7]])
# B = B.T
print(f'initial_pro +>{initial_pro}')
print(f'A => {A}')
print(f'B=> {B}')

x = "ACCDDDDFFCCCCBCFFFCCCCCEDADCCAEFCCCACDDFFCCDDFFCCD"
# x = "AAB"

# a, b = forward(x, A, B, initial_pro)
# print(a.shape)
# print(b)
a, b = backward(x, A, B, initial_pro)
print(a.shape)
print(b)
# tebge slide pish berim
# alpha = forward(O, a, b, initial_distribution)
# print(alpha)