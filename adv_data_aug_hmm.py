# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:06:38 2020

@author: franc
"""
import warnings
from gradient_functions import super_fast_grad_x_lkl#, evaluate_data
import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy import stats, linalg
import time
import sys
import os
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
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



def train(path_train, path_test, path_gt, output_dir, train_size, pca_components=4, w=100, eps=0.05, it_aug=3, max_states=25, adv_method='H', repetitions=1): 

    os.chdir(output_dir)
    os.system(f'mkdir models_train_{train_size}')
    with open(f"log_{train_size}_{pca_components}_{w}_{eps}_{it_aug}_{max_states}_{adv_method}_{repetitions}.txt", 'w+') as log:
        train_size = int(train_size)
        pca_components = int(pca_components)
        w = int(w)
        eps = float(eps)
        it_aug = int(it_aug)
        max_states = int(max_states)
        repetitions = int(repetitions)
        
        log.write("#############################################\n")
        log.write(f"TRAIN SIZE\t{train_size}\n")
        log.write(f"PCA COMPONENTS\t{pca_components}\n")
        log.write(f"WINDOW SIZE\t{w}\n")
        log.write(f"EPSILON\t{eps}\n")
        log.write(f"ITERATION OF AUGMENTATION\t{it_aug}\n")
        log.write(f"MAX N. STATES\t{max_states}\n")
        log.write(f"N. REPETITIONS\t{repetitions}\n")
        log.write("#############################################\n")
        t = True #re-train threshold?
        
        data_train_ = pd.read_csv(path_train).values
        data_test_ = pd.read_csv(path_test).values
        gt = pd.read_csv(path_gt).values[w-1:]
        
        threshs_before = []
        threshs_after = []
        f1s_before = []
        f1s_after = []
        introduced = []
        #attacks_before = []
        #attacks_after = []
        for iteration in range(repetitions):
            print("Repetition #", iteration)
            log.write(f"Repetition #{iteration}\n")
            
            np.random.seed(iteration)
            start = np.random.randint(0, data_train_.shape[0]-train_size)
            print(f"Training set starts at {start} and ends in {start+train_size}")
            log.write(f"Training set starts at {start} and ends in {start+train_size}\n")
            
            data_train = data_train_[start:start+train_size]
            
            if pca_components > 0:
                sc = StandardScaler()
                pca = PCA(n_components = pca_components)
                
                np.random.seed(0)
                data_train = sc.fit_transform(data_train)
                np.random.seed(0)
                data_train = pca.fit_transform(data_train)
                
                data_test = sc.transform(data_test_)
                data_test = pca.transform(data_test_)
            
            else:
                
                data_test = data_test_
                
            ##############n states training##############
            print("Choosing optimal number of states with BIC")
            log.write("Choosing optimal number of states with BIC\n")
            BICs = []
            for K in range(2,max_states):
                np.random.seed(0)
                model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state = 0).fit(data_train)
                ll_train, viterbi_train = model.decode(data_train)
                BICs.append(BIC(data_train.shape[0], K, 2*data_train.shape[1], ll_train))
            
            K = np.argmin(BICs) + 2
            print("N states from BIC", K)
            log.write(f"N states from BIC {K}\n")
            print("Model training")
            log.write("Model training\n")
            start = time.time()
            model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state = 0)#, init_params='')
            np.random.seed(0)
            model.fit(data_train)
            print("Trained in ", time.time()-start)
            log.write(f"Trained in {time.time()-start}\n")
            with open(f"models_train_{train_size}/model_{iteration}_HHAD.pkl", "wb") as file: pickle.dump(model, file)
            
            ##############threshold training##############
            print("Threshold training")
            log.write("Threshold training\n")
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
            log.write(f"Threshold trained in {time.time()-start}\n")
            print("STARTING THRESHOLD", thresh)
            log.write(f"STARTING THRESHOLD {thresh}\n")
            threshs_before.append(thresh)
            
            ##############first evaluation of test set##############
            print("First evaluation on test set")
            log.write("First evaluation on test set\n")
            start = time.time()
            i = w
            scores = []
            re_scores = []
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
                    else:
                        idx_final_score = np.argmax(new_scores)+1
                        re_scores.append(final_score)
                elif adv_method == 'L':
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
                else:
                    print("Adversarial method in input is not recognized")
                    sys.exit()
                ###################################################################
                
                i += 1
            
            answers_blue = np.array(scores) > thresh #first array used after to calculate performances
            
            successful_attacks_test_before = np.bitwise_and(np.array(re_scores)[np.where(gt == 0)[0]] >= thresh, np.array(scores)[np.where(gt == 0)[0]] < thresh)
            print("Successful attacks on test before", sum(successful_attacks_test_before))
            log.write(f"Successful attacks on test before {sum(successful_attacks_test_before)}\n")
            
            ##############adversarial example generation for data augmentation##############
            print("Starting augmentation procedure")
            log.write("Starting augmentation procedure\n")
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
                    elif adv_method == 'L':
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
                    else:
                        print("Adversarial method in input is not recognized")
                        sys.exit()
                    ###################################################################
                    
                    
                    i += 1
                    
                
                adv_positions = np.bitwise_and(np.array(adv_scores) >= thresh, np.array(scores) < thresh)
                print("Number of false positives generated", sum(adv_positions))
                log.write(f"Number of false positives generated {sum(adv_positions)}\n")
                
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
            threshs_after.append(thresh)
            print("Finished in", time.time() - start_time)
            log.write(f"Finished in {time.time() - start_time}\n")
            print("THRESHOLD AFTER AUGMENTATION", thresh)
            log.write("THRESHOLD AFTER AUGMENTATION {thresh}\n")
            with open(f"models_train_{train_size}/model_{iteration}_{adv_method}-AUG.pkl", "wb") as file: pickle.dump(model, file)
            
            ##############re-evaluation on test set with augmented model##############
            print("Re-evaluation of test set with augmented model")
            log.write("Re-evaluation of test set with augmented model\n")
            i = w
            scores = [] #scores must contain the collection of normal scores
            re_scores = [] #re_scores must contain the collection of adversarial scores
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
                    else:
                        idx_final_score = np.argmax(new_scores)+1
                        re_scores.append(final_score)
                elif adv_method == 'L':
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
                else:
                    print("Adversarial method in input is not recognized")
                    sys.exit()
                ###################################################################                
                
                i += 1
            
            re_answers_blue = np.array(scores) > thresh
            
            successful_attacks_test_after = np.bitwise_and(np.array(re_scores)[np.where(gt == 0)[0]] >= thresh, np.array(scores)[np.where(gt == 0)[0]] < thresh)
            print("Successful attacks after", sum(successful_attacks_test_after))
            log.write(f"Successful attacks after {sum(successful_attacks_test_after)}\n")
            
            ##############compute performances##############
            
            conf = confusion_matrix(gt, answers_blue)
            
            TP = conf[0,0]
            TN = conf[1,1]
            FP = conf[0,1]
            FN = conf[1,0]
            
            F1_blue = 2*TP / (2*TP + FP + FN)
            print("F1 before", F1_blue)
            log.write(f"F1 before {F1_blue}\n")
            f1s_before.append(F1_blue)
            
            conf = confusion_matrix(gt, re_answers_blue)
            
            TP = conf[0,0]
            TN = conf[1,1]
            FP = conf[0,1]
            FN = conf[1,0]
            
            F1_blue = 2*TP / (2*TP + FP + FN)
            print("F1 after", F1_blue)
            log.write(f"F1 after {F1_blue}\n")
            f1s_after.append(F1_blue)
            
            log.write("\n")
        
    f1_df = pd.DataFrame()
    f1_df['F1 before'] = f1s_before
    f1_df['F1 after'] = f1s_after
    f1_df.to_csv(f"f1_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    
    thresh_df =  pd.DataFrame()
    thresh_df['Tau before'] = threshs_before
    thresh_df['Tau after'] = threshs_after
    thresh_df.to_csv(f"taus_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)


if __name__ == '__main__':
    
    recognized_adv_methods = {'H':'Hellinger Distance', 'L':'Likelihood'}
    
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    path_gt = sys.argv[3]
    output_dir = sys.argv[4]
    train_sizes = sys.argv[5]
    pca_components = sys.argv[6]
    w = sys.argv[7]
    eps = sys.argv[8]
    it_aug = sys.argv[9]
    max_states = sys.argv[10]
    adv_method = sys.argv[11]
    repetitions = sys.argv[12]
    
    if adv_method not in recognized_adv_methods:
        print("Adversarial method in input is not recognized")
        print("For now the list is", recognized_adv_methods)
        sys.exit()
    
    train_sizes = train_sizes.strip().split(',')
    
    for train_size in train_sizes:
        train(path_train, path_test, path_gt, output_dir, train_size, pca_components, w, eps, it_aug, max_states, adv_method, repetitions)
    
    
    
    
    
    
    