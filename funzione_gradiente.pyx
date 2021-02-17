# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:47:27 2021

@author: franc
"""
cimport numpy as np
import numpy as np
#from scipy import linalg, stats
#from math import e
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] super_fast_grad_x_lkl(double[:,::1] data, int num_states, double[:,::1] means_, double[:,:,::1] covars_, double[::1] startprob_, double[:,::1] transmat_, double[:,::1] ll): #inv_cov_matrix
    cdef Py_ssize_t T = data.shape[0]
    cdef Py_ssize_t D = data.shape[1]
    cdef Py_ssize_t N = num_states
    cdef Py_ssize_t t_star, d_star, t, i, j
    
    cdef double[:,::1] alpha = np.zeros((T, N), dtype=np.double)
    cdef double[:,::1] b = np.exp(ll, dtype=np.double)
    
    #initialization
    for j in range(N):
        alpha[0,j] = startprob_[j]*b[0,j]
    
    #recursion
    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                alpha[t,j] = alpha[t,j] + alpha[t-1,i] * transmat_[i,j] * b[t,j]
   
    cdef double[:,::1] grads = np.zeros((T,D), dtype=np.double)
    cdef double[:,::1] alpha_temp
    cdef double[:,::1] b_temp
    
    for t_star in range(T):
        for d_star in range(D):
            
            alpha_temp = alpha.copy()
            b_temp = b.copy()
            
            for j in range(N):
                b_temp[t_star,j] = b_temp[t_star,j] * ((data[t_star,d_star] - means_[j,d_star]) / (covars_[j,d_star,d_star] + 1e-5 ))
            
            for t in range(t_star, T):
                for j in range(N):
                    if(not t):
                        alpha_temp[0,j] = startprob_[j]*b_temp[0,j]
                    else:
                        alpha_temp[t][j] = 0
                        for i in range(N):
                            alpha_temp[t,j] += alpha_temp[t-1,i] * transmat_[i,j] * b_temp[t,j]
            
            for j in range(N): 
                grads[t_star,d_star] += alpha_temp[T-1,j]
            
    return grads

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,::1] matrix_mult(double[:,::1] first, double[:,::1] second, int r1, int c1, int r2, int c2):
    cdef Py_ssize_t i, j, k
    cdef double[:,::1] result = np.zeros((r1,c2), dtype=np.double)
    
    for i in range(r1):
      for j in range(c2):
         for k in range(c1):
            result[i,j] += first[i,k] * second[k,j]
            
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double hellinger_distance(double[::1] mu_model, double[:,::1] cov_model, double[::1] mu_data, double[:,::1] cov_data):
    cdef Py_ssize_t D = cov_model.shape[0]
    cdef Py_ssize_t i, dim
    
    
    cdef double num_comp1
    cdef double det_model = 1.0
    cdef double det_data = 1.0
    for i in range(D):
        det_model = det_model * cov_model[i,i]
        det_data = det_data * cov_data[i,i]
    
    det_model = det_model ** (0.25)
    det_data = det_data ** (0.25)
    num_comp1 = det_model * det_data
       
    cdef double den_comp1
    cdef double det = 1.0
    for i in range(D):
        det = det * ((cov_data[i,i] + cov_model[i,i]) / 2)
    det = det ** (0.5)
    den_comp1 = det
    
    cdef double comp1 = num_comp1/den_comp1
    
    cdef double comp2
    cdef double[:,::1] diff_means = np.empty((1,D), dtype=np.double)
    cdef double[:,::1] diff_means_T = np.empty((D,1), dtype=np.double)
    cdef double[:,::1] cov_matrix_inverse = np.empty((D,D), dtype=np.double)
    for i in range(D):
        diff_means[0,i] = mu_model[i] - mu_data[i]
        diff_means_T[i,0] = mu_model[i] - mu_data[i]
        cov_matrix_inverse[i,i] = 1. / ((cov_model[i,i] + cov_data[i,i])/2)
    
       
    #comp2 = np.exp((-1/8) * (matrix_mult(diff_means, matrix_mult(cov_matrix_inverse, (diff_means).T, D, D, D, 1), 1, D, D, 1))) 
    comp2 = np.exp((-1./8) * matrix_mult(matrix_mult(diff_means, cov_matrix_inverse, 1, D, D, D), diff_means_T, 1, D, D, 1 )[0,0])
    
    return 1 - (comp1 * comp2)
    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evaluate_data(double[:,::1] data, model, int w):
    cdef Py_ssize_t N = data.shape[0]
    cdef Py_ssize_t D = data.shape[1]
    cdef Py_ssize_t t = w
    cdef Py_ssize_t i
    cdef double[:,::1] mu_model = model.means_
    cdef double[:,:,::1] cov_model = model.covars_
    
    cdef double[::1] scores = np.empty(N-w)
    
    cdef double[::1] mu_data = np.empty(D, dtype=np.double)
    cdef double[:,::1] cov_data = np.empty([D,D], dtype=np.double)
    cdef double somma
    cdef double std_tmp
    
    cdef double[:,::1] Wt
    cdef long[::1] St
    cdef int st, n_occ
    #cdef double[:,::1] X
    
    cdef int maxValue, maxCount, j, count
    for t in range(w,N): 
        #print(t)
        Wt = data[t-w:t]
        St = model.predict(np.asarray(Wt))

        maxValue, maxCount = 0, 0
        for i in range(w):
           count = 0;
           
           for j in range(w):
              if St[j] == St[i]:
                  count = count + 1
           
           if count > maxCount:
              maxCount = count;
              maxValue = St[i];
          
        
        st, n_occ = maxValue, maxCount#stats.mode(St)
        #print(st, n_occ)
        #X = np.empty((n_occ,D), dtype=np.double)
        #a = 0
        #for i in range(w):
        #    if St[i] == st:
        #        X[a] = Wt[i]
        #        a = a + 1
        
        #X = Wt[St == st]

        for dim in range(D):
            somma = 0
            for i in range(w):
                if St[i] == st:
                    somma = somma + Wt[i,dim]
            mu_data[dim] = somma/n_occ#X.shape[0]
        
        for dim in range(D):
            std_tmp = 0
            for i in range(w):
                if St[i] == st:
                    std_tmp = std_tmp + (Wt[i,dim] - mu_data[dim]) ** 2
            cov_data[dim, dim] = std_tmp/(n_occ-1)#(X.shape[0]-1)
            
        score = hellinger_distance(mu_model[st], cov_model[st], mu_data, cov_data)
        scores[t-w] = score
    
    return np.asarray(scores)