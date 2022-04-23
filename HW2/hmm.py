import numpy as np
from numba import jit

def mapping_observation_multiple_seq(observ_lst):
  observation = np.zeros((len(observ_lst), len(observ_lst[0])))
  observation = np.asarray(observation, dtype=np.int8)
  mapping = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5}
  for i in range(len(observ_lst)): 
    for j in range(len(observ_lst[0])):
      observation[i, j] = int(mapping.get(observ_lst[i][j]))
  return observation    

@jit(nopython=True)
def forward(V, a, b, initial_distribution):
    b = b.T
    T = V.shape[1]
    alpha = np.zeros((T, a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    return alpha


@jit(nopython=True) 
def forward_multiple_seq(V, a, b, initial_distribution):
  b = b.T
  R, T = V.shape[0], V.shape[1]
  N = a.shape[0]
  alpha = np.zeros((R, T, a.shape[0]))
  for r in range(R):
    alpha[r, 0, :] = initial_distribution * b[:, V[r, 0]]
  
  for r in range(R):
    for t in range(1, T):
      for j in range(a.shape[0]):
        alpha[r, t, j] = alpha[r, t - 1, :].dot(a[:, j]) * b[j, V[r,t]]
  return alpha
 

@jit(nopython=True)
def backward_multiple_seq(V, a, b):
  b = b.T
  R, T = V.shape[0], V.shape[1]
  beta = np.zeros((R, T, a.shape[0]))
  # setting beta(T) = 1
  for r in range(R):
    beta[r, T-1, :] = np.ones((a.shape[0]))

  # Loop in backward way from T-1 to
  # Due to python indexing the actual loop will be T-2 to 0
  for r in range(R):
    for t in range(T-2, -1, -1):
      for j in range(a.shape[0]):
        beta[r, t, j] = (beta[r, t + 1,:] * b[:, V[r, t+1]]).dot(a[j, :])
  return beta
 

@jit(nopython=True)
def baum_welch_multiple_seq(V, a, b, initial_distribution, n_iter=100):
  N = a.shape[0]
  R, T = V.shape[0], V.shape[1]
  b = b.T

  for n in range(n_iter):
      alpha = forward_multiple_seq(V, a, b, initial_distribution)
      beta = backward_multiple_seq(V, a, b)

      xi = np.zeros((R, N, N, T-1))
      for r in range(R):
        denominator = 0
        for t in range(T-1):
          denominator = np.dot(np.dot(alpha[r, t, :].T, a) * b[:, V[r, t + 1]].T, beta[r, t + 1, :])
          for i in range(N):
              numerator = alpha[r, t, i] * a[i, :] * b[:, V[r, t + 1]].T * beta[r, t + 1, :].T
              xi[r, i, :, t] = numerator / denominator

      gamma = np.sum(xi, axis=2) # sum based on state axis 
      a = np.sum(np.sum(xi, axis=3), axis=0) / np.sum(np.sum(gamma, axis=2), axis=0).reshape((-1, 1))
      gamma_new = np.zeros((R, N, T))
      # Add additional T'th element in gamma
      for r in range(R):
        gamma_new[r] = np.hstack((gamma[r], np.sum(xi[r, :, :, T-2], axis=1).reshape((-1, 1))))
      K = b.shape[1]# unique states
      denominator = np.sum(np.sum(gamma_new, axis=2), axis=0).reshape((-1, 1))

      for l in range(K):
        b[:, l] = np.sum(np.sum(gamma_new[:, :, V[r] == l], axis=2), axis=0)
      b = np.divide(b, denominator.reshape((-1, 1)))
      initial_distribution[:] = np.sum(gamma_new[:, :, 0], axis=0) / R 
  return a, b, initial_distribution


def likelihood(V, a, b, initial_distribution):
    alpha = forward(V, a, b, initial_distribution)
    probability = np.sum(alpha[-1, :])
    return probability 


def readFile(filename):
    fileObj = open(filename, 'r', encoding='utf-8') # open the file in read mode
    words = fileObj.read().splitlines() # puts the file into an array
    fileObj.close()
    return words


def read_init_file(filename):
    initial_pro, A, B = [], [], []
    with open(filename, "r") as f:
        mylist = [line.rstrip('\n') for line in f]
        for i, line in enumerate(mylist):
            if i in range(1, 2):
                initial_pro.append([float(y) for y in line.split()])
            if i in range(4, 10):  
                A.append([float(y) for y in line.split()])
            if i in range(12, 18):    
                B.append([float(y) for y in line.split()])
    initial_pro, A, B = np.asarray(initial_pro), np.asarray(A), np.asarray(B)
    return initial_pro[0], A, B

