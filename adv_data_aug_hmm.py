# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:06:38 2020

@author: franc
"""
import warnings
from funzione_gradiente import super_fast_grad_x_lkl#, evaluate_data
import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy import stats, linalg
import time
import sys
from sklearn.metrics import confusion_matrix
warnings.filterwarnings(action='ignore')

def normalize(a, axis=None):
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum
    
def hellinger_dist(mu_model, cov_model, mu_data, cov_data):
    num_comp1 = (linalg.det(cov_model)**(1/4))*(linalg.det(cov_data)**(1/4))
    den_comp1 = (linalg.det((cov_model + cov_data)/2)**(1/2))
    comp1 = num_comp1/den_comp1
    comp2 = float(np.exp((-1/8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model+cov_data)/2, -1) @ (mu_model - mu_data).T))
    return 1 - comp1 * comp2

def grad_x_hellinger_dist(mu_model, cov_model, data):
    N = data.shape[0]
    mu_data = np.reshape(np.mean(data, axis=0), [1, data.shape[1]])[0]
    cov_data = (np.diag(np.cov(data.T)) + 1e-3) * np.eye(data.shape[1], data.shape[1])
    num_comp1 = (linalg.det(cov_model)**(1/4))*(linalg.det(cov_data)**(1/4))
    den_comp1 = (linalg.det((cov_model + cov_data)/2)**(1/2))
    comp1 = num_comp1/den_comp1
    comp2 = float(np.exp((-1/8) * (mu_model - mu_data) @ np.linalg.matrix_power((cov_model+cov_data)/2, -1) @ (mu_model - mu_data).T))
    grads = np.zeros(data.shape)
    for dim in range(data.shape[1]):
        x_i = data[:,dim]
        der_comp1 = -(np.prod(np.delete(np.diag(cov_data), dim))*((x_i - mu_data[dim])/N)* \
                    (np.prod(np.diag(cov_model)))**(1/4)*0.5*(np.prod(np.diag(cov_data))**(-3/4))* \
                    den_comp1 - 0.5*(num_comp1*((x_i - mu_data[dim])/N)*np.prod(np.delete(np.diag((cov_data+cov_model)/2), dim)))/den_comp1)/ \
                    (den_comp1**2)
        der_comp2 = 0.5*((mu_model[dim]-mu_data[dim])*(cov_model[dim,dim]+cov_data[dim,dim]) - (mu_model[dim]-mu_data[dim])**2*(x_i-mu_data[dim]))/ \
                    (N*((cov_model[dim,dim]+cov_data[dim,dim])**2))*comp2
        grads[:,dim] = der_comp1*comp2 + (-comp1)*der_comp2
    
    return grads

def BIC(size, n_states, n_comp, ll):
    p = n_states**2 + n_comp*n_states - 1
    return -2*ll + p*np.log(size)



def train(path_train, path_test, path_gt, w=100, eps=0.05, it_aug=3, max_states=25, adv_method='H'): 

    w = int(w)
    eps = float(eps)
    it_aug = int(it_aug)
    max_states = int(max_states)
    t = True #re-train threshold?
    
    data_train = pd.read_csv(path_train).values
    data_test = pd.read_csv(path_test).values
    gt = pd.read_csv(path_gt).values[w-1:]
    
    threshs = []
    precs_blue = []
    recs_blue = []
    accs_blue = []
    f1s_blue = []
    fprs_blue = []
    dist_noms_blue = []
    dist_anos_blue = []
    #re_precs_blue = []
    #re_recs_blue = []
    #re_accs_blue = []
    #re_f1s_blue = []
    #re_fprs_blue = []
    #re_dist_noms_blue = []
    #re_dist_anos_blue = []
    #precs_orange = []
    #recs_orange = []
    #accs_orange = []
    #f1s_orange = []
    #fprs_orange = []
    #dist_noms_orange = []
    #dist_anos_orange = []
    #re_precs_orange = []
    #re_recs_orange = []
    #re_accs_orange = []
    #re_f1s_orange = []
    #re_fprs_orange = []
    #re_dist_noms_orange = []
    #re_dist_anos_orange = []
    started = []
    introduced = []
    #attacks_before = []
    #attacks_after = []
    
    ##############n states training##############
    print("Choosing optimal number of states with BIC")
    BICs = []
    for K in range(2,max_states):
        np.random.seed(0)
        model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state = 0).fit(data_train)
        ll_train, viterbi_train = model.decode(data_train)
        BICs.append(BIC(data_train.shape[0], K, 2*data_train.shape[1], ll_train))
    
    K = np.argmin(BICs) + 2
    print("N states from BIC", K)
    print("Model training")
    start = time.time()
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state = 0)#, init_params='')
    np.random.seed(0)
    model.fit(data_train)
    print("Trained in ", time.time()-start)
    
    
    ##############threshold training##############
    print("Threshold training")
    start = time.time()
    i = w
    scores = []
    while(i <= data_train.shape[0]): 
        Wt = data_train[i-w:i].copy()
        ll, St = model.decode(Wt)
        st = stats.mode(St)[0]
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, data_train.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data_train.shape[1], data_test.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        
        scores.append(score)
        i += 1
        
    thresh = np.max(scores) + (1/10*eps)
    print("Threshold trained in ", time.time()-start)
    print("STARTING THRESHOLD", thresh)
    
    ##############first evaluation of test set##############
    print("First evaluation on test set")
    start = time.time()
    i = w
    scores = []
    re_scores = []
    nominal_dist_from_thresh_blue = []
    anomalous_dist_from_thresh_blue = []
    nominal_dist_from_thresh_orange = []
    anomalous_dist_from_thresh_orange = []
    n_w = 0
    while(i <= data_test.shape[0]): #
        current_eps = eps/10
        Wt = data_test[i-w:i].copy()
        ll, St = model.decode(Wt)
        st = stats.mode(St)[0]
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, data_test.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data_test.shape[1], data_test.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        scores.append(score)
        
        ##############CHANGE THIS PART BASED ON GRADIENT USED##############
        if adv_method == "H":
            gradients = grad_x_hellinger_dist(model.means_[st][0], model.covars_[st][0], X)
            signs = np.sign(gradients)
            
            Wt_new = Wt.copy()
            Wt_new[St == st] = Wt_new[St == st] + current_eps*np.array(signs)
            
            prev_score = score
            new_scores = []
            eps_windows = []
            while current_eps <= eps:
                ll_new, St_new = model.decode(Wt_new)
                st_new = stats.mode(St_new)[0]
        
                st_temp = st
                Wt_temp = Wt_new.copy()
                
                X_new = Wt_temp[St_new == st_new]
                new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_test.shape[1]])[0]
                new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_test.shape[1], data_train.shape[1])
                new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                
                changed = False
                while st_new != st_temp and new_score > prev_score:
                    changed = True
                    prev_score = new_score
                    prev_window = Wt_temp.copy()
                    
                    st_temp = st_new
                    
                    other_grads = grad_x_hellinger_dist(model.means_[st_new][0], model.covars_[st_new][0], Wt[St_new == st_new])
                    other_signs = np.sign(other_grads)
                    window_pert = Wt_new.copy()
                    window_pert[St_new == st_new] = window_pert[St_new == st_new] + current_eps*np.array(other_signs)
                    sphere = data_test[i-w:i].copy()
                    sphere[St_new == st_new] = sphere[St_new == st_new] + current_eps*np.array(other_signs)
                    idx_pos = np.where(sphere > 0)
                    idx_neg = np.where(sphere < 0)
                    Wt_temp[idx_pos] = np.minimum(window_pert[idx_pos], sphere[idx_pos])
                    Wt_temp[idx_neg] = np.maximum(window_pert[idx_neg], sphere[idx_neg])
        
                    ll_new, St_new = model.decode(Wt_temp)
                    st_new = stats.mode(St_new)[0]
                    
                    X_new = Wt_temp[St_new == st_new]
                    new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_test.shape[1]])[0]
                    new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_test.shape[1], data_test.shape[1])
                    new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                
                
                if changed and prev_score > new_score:
                    new_score = prev_score
                    Wt_temp = prev_window
                    
                eps_windows.append(Wt_temp)
                
                Wt_new[St == st] = Wt[St == st] + current_eps*np.array(signs)
                current_eps += eps/10
                new_scores.append(new_score)
            
            final_score = np.max(new_scores)
            if final_score < score:
                re_scores.append(score)
                re_score = score
            else:
                idx_final_score = np.argmax(new_scores)+1
                re_scores.append(final_score)
                re_score = final_score
        else:
            ll = model._compute_log_likelihood(Wt)
            gradients = super_fast_grad_x_lkl(Wt, K, model.means_, model.covars_, model.startprob_, model.transmat_, ll) #super_fast_grad_x_lkl(Wt, K, model) #S, 
            signs = np.sign(gradients)
                        
            Wt_new = Wt.copy()
            Wt_new = Wt_new + eps*np.array(signs)
            
            ll_new, St_new = model.decode(Wt_new)
            st_new = stats.mode(St_new)[0]
            X_new = Wt_new[St_new == st_new]
            new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_train.shape[1]])[0]
            new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
            new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
            re_scores.append(new_score)
        ###################################################################
        
        #if gt[n_w] == 0:
        #    nominal_dist_from_thresh_blue.append(thresh-score)
        #    nominal_dist_from_thresh_orange.append(thresh-re_score)
        #else:
        #    anomalous_dist_from_thresh_blue.append(score-thresh)
        #    anomalous_dist_from_thresh_orange.append(re_score-thresh)
        i += 1
        #n_w += 1
    
    #mean_nominal_dist_blue = np.mean(nominal_dist_from_thresh_blue)
    #mean_anomalous_dist_blue = np.mean(anomalous_dist_from_thresh_blue)
    #mean_nominal_dist_orange = np.mean(nominal_dist_from_thresh_orange)
    #mean_anomalous_dist_orange = np.mean(anomalous_dist_from_thresh_orange)
    answers_blue = np.array(scores) > thresh #first array used after to calculate performances
    #answers_orange = np.array(re_scores) > thresh
    
    successful_attacks_test_before = np.bitwise_and(np.array(re_scores)[np.where(gt == 0)[0]] >= thresh, np.array(scores)[np.where(gt == 0)[0]] < thresh)
    print("Successful attacks on test before", sum(successful_attacks_test_before))

    
    ##############adversarial example generation for data augmentation##############
    print("Starting augmentation procedure")
    adv_positions = [1, 1]
    data_augmented = np.concatenate([data_train])
    lengths = [len(data_train)]
    it = 0
    windows_introduced = 0
    start_time = time.time()
    while sum(adv_positions) > 0 and it < it_aug:
        i = w
        scores = []
        final_signs = []
        adv_scores = []
        windows_2 = [] #this variable must be filled with adversarial windows which will be concatenated to data_augmented
    
        while(i <= data_train.shape[0]): #
            current_eps = eps/10
            Wt = data_train[i-w:i].copy()
            ll, St = model.decode(Wt)
            st = stats.mode(St)[0]
            X = Wt[St == st]
            mu = np.reshape(np.mean(X, axis=0), [1, data_train.shape[1]])
            cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
            score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
            scores.append(score)
            ##############CHANGE THIS PART BASED ON GRADIENT USED##############
            if adv_method == 'H':
                gradients = grad_x_hellinger_dist(model.means_[st][0], model.covars_[st][0], X)
                signs = np.sign(gradients)
                
                final_signs.append(signs)
                                
                Wt_new = Wt.copy()
                Wt_new[St == st] = Wt_new[St == st] + current_eps*np.array(signs)
                
                prev_score = score
                new_scores = []
                eps_windows = []
                while current_eps <= eps:
                    ll_new, St_new = model.decode(Wt_new)
                    st_new = stats.mode(St_new)[0]
            
                    st_temp = st
                    Wt_temp = Wt_new.copy()
                    
                    X_new = Wt_temp[St_new == st_new]
                    new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_train.shape[1]])[0]
                    new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
                    new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                    
                    changed = False
                    while st_new != st_temp and new_score > prev_score:
                        changed = True
                        prev_score = new_score
                        prev_window = Wt_temp.copy()
                        
                        st_temp = st_new
                        other_grads = grad_x_hellinger_dist(model.means_[st_new][0], model.covars_[st_new][0], Wt[St_new == st_new])
                        other_signs = np.sign(other_grads)
                        window_pert = Wt_new.copy()
                        window_pert[St_new == st_new] = window_pert[St_new == st_new] + current_eps*np.array(other_signs)
                        sphere = data_train[i-w:i].copy()
                        sphere[St_new == st_new] = sphere[St_new == st_new] + current_eps*np.array(other_signs)
                        idx_pos = np.where(sphere > 0)
                        idx_neg = np.where(sphere < 0)
                        Wt_temp[idx_pos] = np.minimum(window_pert[idx_pos], sphere[idx_pos])
                        Wt_temp[idx_neg] = np.maximum(window_pert[idx_neg], sphere[idx_neg])
            
                        ll_new, St_new = model.decode(Wt_temp)
                        st_new = stats.mode(St_new)[0]
                        
                        X_new = Wt_temp[St_new == st_new]
                        new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_train.shape[1]])[0]
                        new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
                        new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                    
                    
                    if changed and prev_score > new_score:
                        new_score = prev_score
                        Wt_temp = prev_window
                        
                    eps_windows.append(Wt_temp)
                    
                    Wt_new[St == st] = Wt[St == st] + current_eps*np.array(signs)
                    current_eps += eps/10
                    new_scores.append(new_score)
                
                final_score = np.max(new_scores)
                if final_score < score:
                    adv_scores.append(score)
                    windows_2.append(Wt)
                else:
                    idx_final_score = np.argmax(new_scores)+1
                    adv_scores.append(final_score)
                    windows_2.append(eps_windows[idx_final_score-1])
            else:
                ll = model._compute_log_likelihood(Wt)
                gradients = super_fast_grad_x_lkl(Wt, K, model.means_, model.covars_, model.startprob_, model.transmat_, ll) #super_fast_grad_x_lkl(Wt, K, model) #S, 
                signs = np.sign(gradients)
                            
                Wt_new = Wt.copy()
                Wt_new = Wt_new + eps*np.array(signs)
                
                ll_new, St_new = model.decode(Wt_new)
                st_new = stats.mode(St_new)[0]
                X_new = Wt_new[St_new == st_new]
                new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_train.shape[1]])[0]
                new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
                new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                adv_scores.append(new_score)  
                windows_2.append(Wt_new)
            ###################################################################
            
            
            i += 1
            
        
        adv_positions = np.bitwise_and(np.array(adv_scores) >= thresh, np.array(scores) < thresh)
        print("Number of false positives generated", sum(adv_positions))
        
        
        windows_introduced += sum(adv_positions)
        if sum(adv_positions) > 0:
            for pos in np.where(adv_positions)[0]:
                lengths.append(w)
                data_augmented = np.concatenate([data_augmented, windows_2[pos]])
            
            np.random.seed(0)
            model.fit(data_augmented, lengths)
        
            if t:
                ###############threshold re-training###############
                
                i = w
                scores = []
                while(i <= data_train.shape[0]): 
                    Wt = data_train[i-w:i].copy()
                    ll, St = model.decode(Wt)
                    st = stats.mode(St)[0]
                    X = Wt[St == st]
                    mu = np.reshape(np.mean(X, axis=0), [1, data_test.shape[1]])
                    cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data_test.shape[1], data_test.shape[1])
                    score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
                    
                    scores.append(score)
                    i += 1
                    
                #thresh = np.max(scores) + (1/10*eps)
                if np.max(scores) + (1/10*eps) > thresh:
                    thresh = np.max(scores) + (1/10*eps)
                
           
        it += 1
     
    introduced.append(windows_introduced)
    started.append(it)
    threshs.append(thresh)
    print("Finished in", time.time() - start_time)
    print("THRESHOLD AFTER AUGMENTATION", thresh)
    
    ##############re-evaluation on test set with augmented model##############
    print("Re-evaluation of test set with augmented model")
    re_nominal_dist_from_thresh_blue = []
    re_anomalous_dist_from_thresh_blue = []
    re_nominal_dist_from_thresh_orange = []
    re_anomalous_dist_from_thresh_orange = []
    i = w
    scores = [] #scores must contain the collection of normal scores
    re_scores = [] #re_scores must contain the collection of adversarial scores
    n_w = 0
    while(i <= data_test.shape[0]): #
        current_eps = eps/10
        Wt = data_test[i-w:i].copy()
        ll, St = model.decode(Wt)
        st = stats.mode(St)[0]
        X = Wt[St == st]
        mu = np.reshape(np.mean(X, axis=0), [1, data_test.shape[1]])
        cov = (np.diag(np.cov(X.T)) + 1e-5) * np.eye(data_test.shape[1], data_test.shape[1])
        score = hellinger_dist(model.means_[st], model.covars_[st][0], mu, cov)
        scores.append(score)
        
        ##############CHANGE THIS PART BASED ON GRADIENT USED##############
        if adv_method == 'H':
            gradients = grad_x_hellinger_dist(model.means_[st][0], model.covars_[st][0], X)
            signs = np.sign(gradients)
            
            Wt_new = Wt.copy()
            Wt_new[St == st] = Wt_new[St == st] + current_eps*np.array(signs)
            
            prev_score = score
            new_scores = []
            eps_windows = []
            while current_eps <= eps:
                ll_new, St_new = model.decode(Wt_new)
                st_new = stats.mode(St_new)[0]
        
                st_temp = st
                Wt_temp = Wt_new.copy()
                
                X_new = Wt_temp[St_new == st_new]
                new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_test.shape[1]])[0]
                new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_test.shape[1], data_train.shape[1])
                new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                
                changed = False
                while st_new != st_temp and new_score > prev_score:
                    changed = True
                    prev_score = new_score
                    prev_window = Wt_temp.copy()
                    
                    st_temp = st_new
                    #print(i, current_eps)
                    other_grads = grad_x_hellinger_dist(model.means_[st_new][0], model.covars_[st_new][0], Wt[St_new == st_new])
                    other_signs = np.sign(other_grads)
                    window_pert = Wt_new.copy()
                    window_pert[St_new == st_new] = window_pert[St_new == st_new] + current_eps*np.array(other_signs)
                    sphere = data_test[i-w:i].copy()
                    sphere[St_new == st_new] = sphere[St_new == st_new] + current_eps*np.array(other_signs)
                    idx_pos = np.where(sphere > 0)
                    idx_neg = np.where(sphere < 0)
                    Wt_temp[idx_pos] = np.minimum(window_pert[idx_pos], sphere[idx_pos])
                    Wt_temp[idx_neg] = np.maximum(window_pert[idx_neg], sphere[idx_neg])
        
                    ll_new, St_new = model.decode(Wt_temp)
                    st_new = stats.mode(St_new)[0]
                    
                    X_new = Wt_temp[St_new == st_new]
                    new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_test.shape[1]])[0]
                    new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_test.shape[1], data_test.shape[1])
                    new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
                
                
                if changed and prev_score > new_score:
                    new_score = prev_score
                    Wt_temp = prev_window
                    
                eps_windows.append(Wt_temp)
                
                Wt_new[St == st] = Wt[St == st] + current_eps*np.array(signs)
                current_eps += eps/10
                new_scores.append(new_score)
            
            final_score = np.max(new_scores)
            if final_score < score:
                re_scores.append(score)
                re_score = score
            else:
                idx_final_score = np.argmax(new_scores)+1
                re_scores.append(final_score)
                re_score = final_score
        else:
            ll = model._compute_log_likelihood(Wt)
            gradients = super_fast_grad_x_lkl(Wt, K, model.means_, model.covars_, model.startprob_, model.transmat_, ll) #super_fast_grad_x_lkl(Wt, K, model) #S, 
            signs = np.sign(gradients)
                        
            Wt_new = Wt.copy()
            Wt_new = Wt_new + eps*np.array(signs)
            
            ll_new, St_new = model.decode(Wt_new)
            st_new = stats.mode(St_new)[0]
            X_new = Wt_new[St_new == st_new]
            new_mu = np.reshape(np.mean(X_new, axis=0), [1, data_train.shape[1]])[0]
            new_cov = (np.diag(np.cov(X_new.T)) + 1e-5) * np.eye(data_train.shape[1], data_train.shape[1])
            new_score = hellinger_dist(model.means_[st_new], model.covars_[st_new][0], new_mu, new_cov)
            re_scores.append(new_score)
        ###################################################################                
        
        #if gt[n_w] == 0:
        #    re_nominal_dist_from_thresh_blue.append(thresh-score)
        #    re_nominal_dist_from_thresh_orange.append(thresh-re_score)
        #else:
        #    re_anomalous_dist_from_thresh_blue.append(score-thresh)
        #    re_anomalous_dist_from_thresh_orange.append(re_score-thresh)
        i += 1
        #n_w += 1
    
    #re_mean_nominal_dist_blue = np.mean(re_nominal_dist_from_thresh_blue)
    #print("Mean nominal distance before", mean_nominal_dist_blue, "\nMean nominal distance after", re_mean_nominal_dist_blue)    
    #re_mean_anomalous_dist_blue = np.mean(re_anomalous_dist_from_thresh_blue)
    #print("Mean anomalous distance before", mean_anomalous_dist_blue, "\nMean anomalous distance after", re_mean_anomalous_dist_blue)   
    #re_mean_nominal_dist_orange = np.mean(re_nominal_dist_from_thresh_orange)
    #print("Mean nominal distance before", mean_nominal_dist_orange, "\nMean nominal distance after", re_mean_nominal_dist_orange)    
    #re_mean_anomalous_dist_orange = np.mean(re_anomalous_dist_from_thresh_orange)
    #print("Mean anomalous distance before", mean_anomalous_dist_orange, "\nMean anomalous distance after", re_mean_anomalous_dist_orange)   
    re_answers_blue = np.array(scores) > thresh
    #re_answers_orange = np.array(re_scores) > thresh
    
    successful_attacks_test_after = np.bitwise_and(np.array(re_scores)[np.where(gt == 0)[0]] >= thresh, np.array(scores)[np.where(gt == 0)[0]] < thresh)
    print("Successful attacks after", sum(successful_attacks_test_after))
    
    ##############compute performances##############
    
    conf = confusion_matrix(gt, answers_blue)
    
    TP = conf[0,0]
    TN = conf[1,1]
    FP = conf[0,1]
    FN = conf[1,0]
    
    precision_blue = TP / (TP + FP)
    recall_blue = TP / (TP + FN)
    accuracy_blue = (TP + TN) / (TP + TN + FP + FN)
    F1_blue = 2*TP / (2*TP + FP + FN)
    FPR_blue = FP / (FP + TN)
    print("F1 before", F1_blue)
    
    #precs_blue.append(precision_blue)
    #recs_blue.append(recall_blue)
    #accs_blue.append(accuracy_blue)
    #f1s_blue.append(F1_blue)
    #fprs_blue.append(FPR_blue)
    #dist_noms_blue.append(mean_nominal_dist_blue)
    #dist_anos_blue.append(mean_anomalous_dist_blue)
    
    
    conf = confusion_matrix(gt, re_answers_blue)
    
    TP = conf[0,0]
    TN = conf[1,1]
    FP = conf[0,1]
    FN = conf[1,0]
    
    precision_blue = TP / (TP + FP)
    recall_blue = TP / (TP + FN)
    accuracy_blue = (TP + TN) / (TP + TN + FP + FN)
    F1_blue = 2*TP / (2*TP + FP + FN)
    FPR_blue = FP / (FP + TN)
    print("F1 after", F1_blue)
    
    #re_precs_blue.append(precision_blue)
    #re_recs_blue.append(recall_blue)
    #re_accs_blue.append(accuracy_blue)
    #re_f1s_blue.append(F1_blue)
    #re_fprs_blue.append(FPR_blue)
    #re_dist_noms_blue.append(re_mean_nominal_dist_blue)
    #re_dist_anos_blue.append(re_mean_anomalous_dist_blue)
    
    #attacks_before.append(sum(successful_attacks_test_before))
    #attacks_after.append(sum(successful_attacks_test_after))
    
    '''
    conf = confusion_matrix(gt, answers_orange)
    print("Before\n", conf)
    
    TP = conf[0,0]
    TN = conf[1,1]
    FP = conf[0,1]
    FN = conf[1,0]
    
    precision_orange = TP / (TP + FP)
    recall_orange = TP / (TP + FN)
    accuracy_orange = (TP + TN) / (TP + TN + FP + FN)
    F1_orange = 2*TP / (2*TP + FP + FN)
    FPR_orange = FP / (FP + TN)
    print("Precision", precision_orange, "Recall", recall_orange, "Accuracy", accuracy_orange, "F1", F1_orange, "FPR", FPR_orange)
    
    precs_orange.append(precision_orange)
    recs_orange.append(recall_orange)
    accs_orange.append(accuracy_orange)
    f1s_orange.append(F1_orange)
    fprs_orange.append(FPR_orange)
    #dist_noms_orange.append(mean_nominal_dist_orange)
    #dist_anos_orange.append(mean_anomalous_dist_orange)
    
    
    conf = confusion_matrix(gt, re_answers_orange)
    print("After\n", conf)
    
    TP = conf[0,0]
    TN = conf[1,1]
    FP = conf[0,1]
    FN = conf[1,0]
    
    precision_orange = TP / (TP + FP)
    recall_orange = TP / (TP + FN)
    accuracy_orange = (TP + TN) / (TP + TN + FP + FN)
    F1_orange = 2*TP / (2*TP + FP + FN)
    FPR_orange = FP / (FP + TN)
    print("Precision", precision_orange, "Recall", recall_orange, "Accuracy", accuracy_orange, "F1", F1_orange, "FPR", FPR_orange)
    
    re_precs_orange.append(precision_orange)
    re_recs_orange.append(recall_orange)
    re_accs_orange.append(accuracy_orange)
    re_f1s_orange.append(F1_orange)
    re_fprs_orange.append(FPR_orange)
    #re_dist_noms_orange.append(re_mean_nominal_dist_orange)
    #re_dist_anos_orange.append(re_mean_anomalous_dist_orange)
    
    #attacks_before.append(sum(successful_attacks_test_before))
    #attacks_after.append(sum(successful_attacks_test_after))
    '''
    '''
    ##############store performances##############
    if t:        
        np.savetxt('prec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', precs_blue)
        np.savetxt('rec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', recs_blue)
        np.savetxt('acc_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', accs_blue)
        np.savetxt('f1_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', f1s_blue)
        np.savetxt('fpr_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', fprs_blue)
        np.savetxt('dist_nom_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', dist_noms_blue)
        np.savetxt('dist_ano_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', dist_anos_blue)
        np.savetxt('started_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', started)
        np.savetxt('introduced_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', introduced)
        np.savetxt('attacks_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', attacks_before)
        
        np.savetxt('re_prec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_precs_blue)
        np.savetxt('re_rec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_recs_blue)
        np.savetxt('re_acc_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_accs_blue)
        np.savetxt('re_f1_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_f1s_blue)
        np.savetxt('re_fpr_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_fprs_blue)
        np.savetxt('re_dist_nom_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_dist_noms_blue)
        np.savetxt('re_dist_ano_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_dist_anos_blue)
        np.savetxt('re_attacks_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', attacks_after)
        np.savetxt('thresh_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', threshs)
        
        
        np.savetxt('prec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', precs_orange)
        np.savetxt('rec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', recs_orange)
        np.savetxt('acc_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', accs_orange)
        np.savetxt('f1_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', f1s_orange)
        np.savetxt('fpr_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', fprs_orange)
        np.savetxt('dist_nom_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', dist_noms_orange)
        np.savetxt('dist_ano_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', dist_anos_orange)
        np.savetxt('started_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', started)
        np.savetxt('introduced_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', introduced)
        np.savetxt('attacks_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', attacks_before)
        
        np.savetxt('re_prec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_precs_orange)
        np.savetxt('re_rec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_recs_orange)
        np.savetxt('re_acc_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_accs_orange)
        np.savetxt('re_f1_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_f1s_orange)
        np.savetxt('re_fpr_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_fprs_orange)
        np.savetxt('re_dist_nom_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_dist_noms_orange)
        np.savetxt('re_dist_ano_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', re_dist_anos_orange)
        np.savetxt('re_attacks_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'_t.txt', attacks_after)
        
    else:
        np.savetxt('prec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', precs_blue)
        np.savetxt('rec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', recs_blue)
        np.savetxt('acc_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', accs_blue)
        np.savetxt('f1_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', f1s_blue)
        np.savetxt('fpr_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', fprs_blue)
        np.savetxt('dist_nom_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', dist_noms_blue)
        np.savetxt('dist_ano_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', dist_anos_blue)
        np.savetxt('started_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', started)
        np.savetxt('introduced_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', introduced)
        np.savetxt('attacks_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', attacks_before)
        
        np.savetxt('re_prec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_precs_blue)
        np.savetxt('re_rec_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_recs_blue)
        np.savetxt('re_acc_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_accs_blue)
        np.savetxt('re_f1_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_f1s_blue)
        np.savetxt('re_fpr_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_fprs_blue)
        np.savetxt('re_dist_nom_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_dist_noms_blue)
        np.savetxt('re_dist_ano_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_dist_anos_blue)
        np.savetxt('re_attacks_blue_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', attacks_after)
        np.savetxt('thresh_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', threshs)
        
        np.savetxt('prec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', precs_orange)
        np.savetxt('rec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', recs_orange)
        np.savetxt('acc_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', accs_orange)
        np.savetxt('f1_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', f1s_orange)
        np.savetxt('fpr_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', fprs_orange)
        np.savetxt('dist_nom_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', dist_noms_orange)
        np.savetxt('dist_ano_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', dist_anos_orange)
        np.savetxt('started_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', started)
        np.savetxt('introduced_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', introduced)
        np.savetxt('attacks_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', attacks_before)
        
        np.savetxt('re_prec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_precs_orange)
        np.savetxt('re_rec_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_recs_orange)
        np.savetxt('re_acc_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_accs_orange)
        np.savetxt('re_f1_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_f1s_orange)
        np.savetxt('re_fpr_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_fprs_orange)
        np.savetxt('re_dist_nom_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_dist_noms_orange)
        np.savetxt('re_dist_ano_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', re_dist_anos_orange)
        np.savetxt('re_attacks_orange_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', attacks_after)
        np.savetxt('thresh_'+str(train_size)+'_'+str(it_aug)+'_'+str(max_states)+'.txt', threshs)
    '''


if __name__ == '__main__':
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    path_gt = sys.argv[3]
    w = sys.argv[4]
    eps = sys.argv[5]
    it_aug = sys.argv[6]
    max_states = sys.argv[7]
    adv_method = sys.argv[8]
    
    train(path_train, path_test, path_gt, w, eps, it_aug, max_states, adv_method)
    
    
    
    
    
    
    