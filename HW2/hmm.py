# implement hmm model
import numpy as np
# O = input columns : 51 in this case
mapping = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5}

# def forward(O, A, B, initial_probability):
#     T = len(O) # time steps
#     N = A.shape[0] # number of states
#     alpha = np.zeros((N, T))
#     for state in range(0, N):
#         alpha[state, 0] = initial_probability[0][state] * B[mapping.get(O[0]), state] 
#         # alpha[state, 0] = initial_probability[0][state] 
#     for time in range(1, T):
#         for state in range(0, N):
#             alpha[state, time] = alpha[:, time-1].dot(A[state, :] * B[mapping.get(O[time]), state]) 
#     termination_step =  np.sum(alpha[:, -1], axis=0)       
#     return alpha, termination_step   

def forward(O, A, B, initial_probability):
    T = len(O) # time steps
    N = A.shape[0] # number of states
    B = B.T
    alpha = np.zeros((T, N))
    for state in range(0, N):
        alpha[0, state] = initial_probability[0][state] * B[state, mapping.get(O[0])]
    for time in range(1, T):
        for state in range(0, N):
            alpha[time, state] = alpha[time-1, :].dot(A[:, state] * B[state, mapping.get(O[time])]) 
    termination_step =  np.sum(alpha[T-1, :])       
    return alpha, termination_step      

def backward(O, A, B, initial_probability):
    T = len(O)
    N = A.shape[0]
    beta = np.zeros((T, N))
    for state in range(0, N):
        beta[-1, state] = 1
    for time in range(T-2, -1, -1):
        for state in range(0, N):
            # beta[time, state] = (A[state, :] * B[mapping.get(O[time+1]),:]).dot(beta[:, time+1]) 
            beta[time, state] = (beta[time+1, :]*B[:, mapping.get(O[time+1])]).dot(A[state, :] )       
    termination_step = np.sum(((beta[0, :])*(initial_probability[0]).dot(B[:, mapping.get(O[0])])))   
    return beta, termination_step   

def baum_welch(O, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(O)
 
    for n in range(n_iter):
        alpha = forward(O, a, b, initial_distribution)
        beta = backward(O, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, mapping.get(O[t + 1])].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, mapping.get(O[t + 1])].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, mapping.get(O) == l], axis=1)
 
        b = np.divide(b, denominator.reshape((-1, 1)))
 
    return {"a":a, "b":b}

# def forward(V, a, b, initial_distribution):
#     alpha = np.zeros((V.shape[0], a.shape[0]))
#     alpha[0, :] = initial_distribution * b[:, V[0]]
 
#     for t in range(1, V.shape[0]):
#         for j in range(a.shape[0]):
#             # Matrix Computation Steps
#             #                  ((1x2) . (1x2))      *     (1)
#             #                        (1)            *     (1)
#             alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
#     termination_step =  np.sum(alpha[-1, :], axis=1) 
#     return alpha, termination_step
 
 
# def backward(V, a, b, initial_probability):
#     beta = np.zeros((V.shape[0], a.shape[0]))
 
#     # setting beta(T) = 1
#     beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
#     # Loop in backward way from T-1 to
#     # Due to python indexing the actual loop will be T-2 to 0
#     for t in range(V.shape[0] - 2, -1, -1):
#         for j in range(a.shape[0]):
#             beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
#     termination_step = np.sum(((beta[:, 0])*(initial_probability[0]).dot(b[O[0], :])), axis=0)
#     return beta, termination_step