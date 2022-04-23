import sys
import numpy as np
from hmm import *
# from hmmlearn import hmm

num_iteration = int(sys.argv[1])
initial_model_addres = sys.argv[2]
observations = sys.argv[3]
result_addres = sys.argv[4]
# print(num_iteration, initial_model_addres, observations, result_addres)

initial_pro, A, B = read_init_file(initial_model_addres)
observ = mapping_observation_multiple_seq(readFile(observations))
# print(observ.shape)
new_A, new_B, new_pi = baum_welch_multiple_seq(observ, A, B, initial_pro, num_iteration)

print(f'new A => \n{new_A}')
print(f'new A => \n{new_A.shape}')
print(f'new B => \n{new_B.T}')
print(f'new B => \n{new_B.shape}')
print(f'new pi => \n{new_pi}')
print(f'new pi => \n{new_pi.shape}')


