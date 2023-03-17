# -*- coding: utf-8 -*-
"""
This software is the implementation of the following article submitted to TPAMI:
	Castellini A., Masillo F., Azzalini D., Amigoni F., Farinelli A., Adversarial Data Augmentation for HMM-based Anomaly Detection
    
In this stage, the software is intended for reviewers' use only.
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
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import mse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from keras import backend as bk
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error
import tsaug
from sklearn import metrics
import matplotlib.pyplot as plt
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



def train(path_train, path_test, path_gt, output_dir, train_size, pca_components=4, w=100, eps=0.05, it_aug=3, max_states=25, adv_method='H', competitors = ['D', 'O'], repetitions=1): 

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
        gt_ = pd.read_csv(path_gt).values
        
        threshs_before = []
        threshs_after = []
        precision_before = []
        precision_after = []
        recall_before = []
        recall_after = []
        accuracy_before = []
        accuracy_after = []
        f1s_before = []
        f1s_after = []
        kappa_before = []
        kappa_after = []
        introduced = []
        fprs_before = []
        tprs_before = []
        fprs_after = []
        tprs_after = []
        attacks_before = []
        attacks_after = []
        competitors_performances = dict()
        for competitor in competitors:
            competitors_performances[competitor] = dict()
            competitors_performances[competitor]['precision'] = []
            competitors_performances[competitor]['recall'] = []
            competitors_performances[competitor]['accuracy'] = []
            competitors_performances[competitor]['F1'] = []
            competitors_performances[competitor]['kappa'] = []
        for iteration in range(repetitions):
            print("Repetition #", iteration)
            log.write(f"Repetition #{iteration}\n")
            
            #np.random.seed((iteration+repetitions)*7)
            np.random.seed(iteration)
            start = np.random.randint(0, data_train_.shape[0]-train_size)
            print(f"Training set starts at {start} and ends in {start+train_size}")
            log.write(f"Training set starts at {start} and ends in {start+train_size}\n")
            
            
            
            if pca_components > 0:
                data_train = data_train_[start:start+train_size]
                print("data_train", data_train.shape)
                sc = StandardScaler()
                pca = PCA(n_components = pca_components)
                
                np.random.seed(0)
                data_train = sc.fit_transform(data_train)
                np.random.seed(0)
                data_train = pca.fit_transform(data_train)
                
                data_test = sc.transform(data_test_)
                data_test = pca.transform(data_test)
                
                gt = gt_[w-1:]
            else:
                print("No PCA performed")
                data_train = data_train_[start:start+train_size]
                data_test = data_test_
                gt = gt_[w-1:]
            
            ##############n states training##############
            print("Choosing optimal number of states with BIC")
            log.write("Choosing optimal number of states with BIC\n")
            pathModel = f"models_train_{train_size}/model_{iteration}_HHAD.pkl"
            if not os.path.exists(pathModel):
                print("Performing BIC")
                BICs = []
                for K in range(2,max_states):
                    try:
                        np.random.seed(0)
                        model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=100, random_state = 0).fit(data_train)
                        ll_train, viterbi_train = model.decode(data_train)
                        BICs.append(BIC(data_train.shape[0], K, 2*data_train.shape[1], ll_train))
                    except:
                        BICs.append(np.inf)
                
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
            else:
                with open(pathModel, "rb") as file: 
                    model = pickle.load(file)
                K = model.n_components
                print("Model loaded with", K, "states")
                log.write(f"Model loaded with {K} states\n")
            
            
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
            
            print("Computing successful attacks")
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
            log.write(f"THRESHOLD AFTER AUGMENTATION {thresh}\n")
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
            
            
            ##############anomaly detection with competitors##############
            for competitor in competitors:
                ###OTHER DETECTORS###
                if competitor == 'A':
                    print('AUTOENCORER-VANILLA')
                    ae_train = []
                    
                    i = w
                    while(i <= data_train.shape[0]):
                        ae_train.append((data_train[i-w:i].copy()).flatten())
                        i += 1
                
                    ae_train = np.array(ae_train)
                    
                    input_ts = keras.Input(shape=(w*data_train.shape[1],))
                    encoded = layers.Dense(128, activation='relu')(input_ts)
                    encoded = layers.Dense(64, activation='relu')(encoded)
                    encoded = layers.Dense(32, activation='relu')(encoded)
                    decoded = layers.Dense(64, activation='relu')(encoded)
                    decoded = layers.Dense(128, activation='relu')(decoded)
                    decoded = layers.Dense(w*data_train.shape[1], activation='linear')(decoded)
                
                    autoencoder = keras.Model(input_ts, decoded)
                    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
                    
                    autoencoder.fit(ae_train, ae_train, epochs=100, batch_size=32, shuffle=True, verbose=0)
                
                    ##############threshold training##############
                    #print("Threshold training")
                    reconstructed_train = autoencoder.predict(ae_train)
                    
                    reconstruction_errors = [mean_squared_error(ae_train[i], reconstructed_train[i]) for i in range(len(ae_train))]
                    thresh = np.max(reconstruction_errors) + (1/10*eps)
                    
                    i = w
                    ae_test = []
                    while(i <= data_test.shape[0]): #
                        Wt = data_test[i-w:i].copy().flatten()
                        ae_test.append(Wt)#.reshape((1,400))
                
                        i += 1
                
                    ae_test = np.array(ae_test)
                    reconstructed_test = autoencoder.predict(ae_test, verbose=1)
                    
                    scores = [mean_squared_error(ae_test[i], reconstructed_test[i]) for i in range(len(ae_test))]
                    answers_competitor = np.array(scores) > thresh #first array used after to calculate performances
                    
                    ##############compute performances##############
                
                    conf = confusion_matrix(gt, answers_competitor)
                
                    TP = conf[0,0]
                    TN = conf[1,1]
                    FP = conf[0,1]
                    FN = conf[1,0]
                    
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    F1 = 2*TP / (2*TP + FP + FN)
                    kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
                    
                    competitors_performances[competitor]['precision'].append(precision)
                    competitors_performances[competitor]['recall'].append(recall)
                    competitors_performances[competitor]['accuracy'].append(accuracy)
                    competitors_performances[competitor]['F1'].append(F1)
                    competitors_performances[competitor]['kappa'].append(kappa)
                    
                    continue
                elif competitor == 'M':
                    print("AUTOENCODER LSTM")
                    ae_train = []
                    
                    i = w
                    while(i <= data_train.shape[0]):
                        ae_train.append(data_train[i-w:i].copy())
                        i += 1
                
                    ae_train = np.array(ae_train)

                    LSTM_model = Sequential()
                    LSTM_model.add(LSTM(128, activation='tanh', input_shape=(w,data_train.shape[1]), return_sequences=True))
                    LSTM_model.add(LSTM(64, activation='tanh', return_sequences=False))
                    LSTM_model.add(RepeatVector(w))
                    LSTM_model.add(LSTM(64, activation='tanh', return_sequences=True))
                    LSTM_model.add(LSTM(128, activation='tanh', return_sequences=True))
                    LSTM_model.add(TimeDistributed(Dense(data_train.shape[1])))
                    
                    LSTM_model.compile(optimizer='adam', loss='mse')
                    
                    LSTM_model.fit(ae_train, ae_train, epochs=30, batch_size=128, verbose=0)
                
                    reconstructed_train = LSTM_model.predict(ae_train)
                    reconstruction_errors = [np.mean(mean_squared_error(ae_train[i], reconstructed_train[i])) for i in range(len(ae_train))]
                    thresh = np.max(reconstruction_errors) + (1/10*eps)
                        
                    ae_test= []
                    
                    i = w
                    while(i <= data_test.shape[0]):
                        ae_test.append((data_test[i-w:i].copy()))
                        i += 1
                
                    ae_test = np.array(ae_test)
                    
                    reconstructed_test = LSTM_model.predict(ae_test)

                    scores = [np.mean(mean_squared_error(ae_test[i], reconstructed_test[i])) for i in range(len(ae_test))]
                    
                    answers_competitor = np.array(scores) > thresh #first array used after to calculate performances
                    
                    ##############compute performances##############
                
                    conf = confusion_matrix(gt, answers_competitor.reshape((-1, 1)))
                
                    TP = conf[0,0]
                    TN = conf[1,1]
                    FP = conf[0,1]
                    FN = conf[1,0]
                    
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    F1 = 2*TP / (2*TP + FP + FN)
                    kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
                    
                    competitors_performances[competitor]['precision'].append(precision)
                    competitors_performances[competitor]['recall'].append(recall)
                    competitors_performances[competitor]['accuracy'].append(accuracy)
                    competitors_performances[competitor]['F1'].append(F1)
                    competitors_performances[competitor]['kappa'].append(kappa)
                    
                    continue
                elif competitor == 'O':
                    print("ONECLASS-SVM")
                    ae_train = []
                    
                    i = w
                    while(i <= data_train.shape[0]):
                        ae_train.append((data_train[i-w:i].copy()).flatten())
                        i += 1

                    ae_train = np.array(ae_train)
                    
                    one_class_svm = OneClassSVM(nu=0.0001, kernel = 'rbf', gamma = 'auto').fit(ae_train)
                    
                    i = w
                    ae_test = []
                    while(i <= data_test.shape[0]): #
                        Wt = data_test[i-w:i].copy().flatten()
                        ae_test.append(Wt)
                        i += 1
                
                    prediction = one_class_svm.predict(ae_test)
                    # Change the anomalies' values to make it consistent with the true values
                    prediction = [1 if i==-1 else 0 for i in prediction]
                    
                    answers_competitor = prediction #first array used after to calculate performances
                    
                    conf = confusion_matrix(gt, answers_competitor)
                
                    TP = conf[0,0]
                    TN = conf[1,1]
                    FP = conf[0,1]
                    FN = conf[1,0]
                    
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    F1 = 2*TP / (2*TP + FP + FN)
                    kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
                    
                    competitors_performances[competitor]['precision'].append(precision)
                    competitors_performances[competitor]['recall'].append(recall)
                    competitors_performances[competitor]['accuracy'].append(accuracy)
                    competitors_performances[competitor]['F1'].append(F1)
                    competitors_performances[competitor]['kappa'].append(kappa)
                    continue
                
                ###AUGMENTERS###
                windows_introduced = 0
                data_augmented = np.concatenate([data_train])
                lengths = [len(data_train)]

                i = w
                                
                if competitor == 'S':
                    print("SMOTE")
                    windows = []
                    while(i <= data_train.shape[0]):
                        Wt = data_train[i-w:i].copy()
                        windows.append(Wt.flatten())
                        i+=1
                    ratio_number_of_new_windows = 0.05
                    windows_extended = windows + [windows[0]]*(int(np.ceil(len(windows)*ratio_number_of_new_windows))+len(windows))
                    gt_windows = np.zeros(len(windows_extended))
                    gt_windows[len(windows)+1:] = 1
                    oversampler = SMOTE(sampling_strategy = 'not majority')
                    windows_resampled, gt_windows_resampled = oversampler.fit_resample(windows_extended, gt_windows)
                    
                    for i_window in range(len(windows_extended), len(windows_resampled)):
                        lengths.append(w)
                        Wt_new = np.reshape(windows_resampled[i_window], (-1,data_train.shape[1]))
                        data_augmented = np.concatenate([data_augmented, Wt_new])
                else: 
                    while(i <= data_train.shape[0]): #        
                        if i%20 == 0:
                            Wt = data_train[i-w:i].copy()
            
                            Wt_tmp = Wt.copy()
            
                            if competitor == 'D':
                                #print("DRIFT")
                                Wt_new = np.array([tsaug.Drift(max_drift=0.05, n_drift_points=5).augment(Wt_tmp.T[dimension]) for dimension in range(data_train.shape[1])]).T
                            elif competitor == 'R':
                                #print("UNIFORM")
                                Wt_new = np.array([tsaug.AddNoise(distr='uniform', scale=0.02).augment(Wt_tmp.T[dimension]) for dimension in range(data_train.shape[1])]).T
                            elif competitor == 'G':
                                #print("GAUSSIAN")
                                Wt_new = np.array([tsaug.AddNoise(scale=0.02).augment(Wt_tmp.T[dimension]) for dimension in range(data_train.shape[1])]).T
                            else:
                                exit()
            
                            lengths.append(w)    
                            data_augmented = np.concatenate([data_augmented, Wt_new])
                            windows_introduced = windows_introduced + 1
                            
                        i += 1
                        
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
                                
                    #if np.max(scores) + (1/10*eps) > thresh:
                    thresh = np.max(scores) + (1/10*eps)
                            
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
                    
                    i += 1
                
                answers_competitor = np.array(scores) > thresh
                conf = confusion_matrix(gt, answers_competitor)
                
                TP = conf[0,0]
                TN = conf[1,1]
                FP = conf[0,1]
                FN = conf[1,0]
                
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                F1 = 2*TP / (2*TP + FP + FN)
                kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
                
                competitors_performances[competitor]['precision'].append(precision)
                competitors_performances[competitor]['recall'].append(recall)
                competitors_performances[competitor]['accuracy'].append(accuracy)
                competitors_performances[competitor]['F1'].append(F1)
                competitors_performances[competitor]['kappa'].append(kappa)
            
            ##############compute performances##############
            
            conf = confusion_matrix(gt, answers_blue)
            
            TP = conf[0,0]
            TN = conf[1,1]
            FP = conf[0,1]
            FN = conf[1,0]
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            F1_blue = 2*TP / (2*TP + FP + FN)
            kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
            print("F1 before", F1_blue)
            log.write(f"F1 before {F1_blue}\n")
            precision_before.append(precision)
            recall_before.append(recall)
            accuracy_before.append(accuracy)
            f1s_before.append(F1_blue)
            kappa_before.append(kappa)
            
            conf = confusion_matrix(gt, re_answers_blue)
            
            TP = conf[0,0]
            TN = conf[1,1]
            FP = conf[0,1]
            FN = conf[1,0]
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            F1_blue = 2*TP / (2*TP + FP + FN)
            kappa = (2*(TP*TN - FN*FP)) / ((TP+FP)*(FP+TN) + (TP+FN)*(FN+TN))
            print("F1 after", F1_blue)
            log.write(f"F1 after {F1_blue}\n")
            precision_after.append(precision)
            recall_after.append(recall)
            accuracy_after.append(accuracy)
            f1s_after.append(F1_blue)
            kappa_after.append(kappa)
            
            log.write("\n")
    
    precision_df = pd.DataFrame()
    precision_df['HHAD'] = precision_before
    precision_df[f'{adv_method}-AUG'] = precision_after
    for competitor in competitors:
        precision_df[f'{competitor}'] = competitors_performances[competitor]['precision']
    precision_df.to_csv(f"precision_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)

    recall_df = pd.DataFrame()
    recall_df['HHAD'] = recall_before
    recall_df[f'{adv_method}-AUG'] = recall_after
    for competitor in competitors:
        recall_df[f'{competitor}'] = competitors_performances[competitor]['recall']
    recall_df.to_csv(f"recall_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)

    accuracy_df = pd.DataFrame()
    accuracy_df['HHAD'] = accuracy_before
    accuracy_df[f'{adv_method}-AUG'] = accuracy_after
    for competitor in competitors:
        accuracy_df[f'{competitor}'] = competitors_performances[competitor]['accuracy']
    accuracy_df.to_csv(f"accuracy_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)

    f1_df = pd.DataFrame()
    f1_df['HHAD'] = f1s_before
    f1_df[f'{adv_method}-AUG'] = f1s_after
    for competitor in competitors:
        f1_df[f'{competitor}'] = competitors_performances[competitor]['F1']
    f1_df.to_csv(f"f1_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    
    kappa_df = pd.DataFrame()
    kappa_df['HHAD'] = kappa_before
    kappa_df[f'{adv_method}-AUG'] = kappa_after
    for competitor in competitors:
        kappa_df[f'{competitor}'] = competitors_performances[competitor]['kappa']
    kappa_df.to_csv(f"kappa_scores_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    
    thresh_df =  pd.DataFrame()
    thresh_df['Tau before'] = threshs_before
    thresh_df['Tau after'] = threshs_after
    thresh_df.to_csv(f"taus_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    
    '''
    fpr_df = pd.DataFrame(fprs_before, columns=['FPR before'])
    fpr_df.to_csv(f"fpr_before_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    fpr_df = pd.DataFrame(fprs_after, columns=['FPR after'])
    fpr_df.to_csv(f"fpr_after_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    
    tpr_df = pd.DataFrame(tprs_before, columns=['TPR before'])
    tpr_df.to_csv(f"tpr_before_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    tpr_df = pd.DataFrame(tprs_after, columns=['TPR after'])
    tpr_df.to_csv(f"tpr_after_train_train_size_{train_size}_adv_method_{adv_method}.csv", index=False)
    '''

from multiprocessing import Pool
if __name__ == '__main__':
    
    recognized_adv_methods = {'H':'Hellinger Distance', 'L':'Likelihood'}
    
    recognized_competitors = {'D': 'Drift-augmenter', 'R': 'Uniform-augmenter', 'G': 'Gaussian-augmenter', 'S': 'SMOTE', 
                              'A': 'AutoEncoder-vanilla', 'M': 'LSTM-AutoEncoder', 'O': 'OneClass-SVM'}
    
    parameter_list = sys.argv
    
    if "--train" in parameter_list:
        path_train = os.path.abspath(sys.argv[parameter_list.index("--train")+1])
    else:
        print("Mandatory parameter --train not found please check input parameters")
        sys.exit()
    if "--test" in parameter_list:
        path_test = os.path.abspath(sys.argv[parameter_list.index("--test")+1])
    else:
        print("Mandatory parameter --test not found please check input parameters")
        sys.exit()
    if "--gt" in parameter_list:
        path_gt = os.path.abspath(sys.argv[parameter_list.index("--gt")+1])
    else:
        print("Mandatory parameter --gt not found please check input parameters")
        sys.exit()
    if "--output_dir" in parameter_list:
        output_dir = os.path.abspath(sys.argv[parameter_list.index("--output_dir")+1])
    else:
        print("Mandatory parameter --output_dir not found please check input parameters")
        sys.exit()
    if "--train_sizes" in parameter_list:
        train_sizes = sys.argv[parameter_list.index("--train_sizes")+1]
    else:
        print("Mandatory parameter --train_sizes not found please check input parameters")
        sys.exit()
    if "--pca" in parameter_list:
        pca_components = sys.argv[parameter_list.index("--pca")+1]
    else:
        pca_components = 4
    if "--w" in parameter_list:
        w = sys.argv[parameter_list.index("--w")+1]
    else:
        w = 100
    if "--eps" in parameter_list:
        eps = sys.argv[parameter_list.index("--eps")+1]
    else:
        eps = 0.05
    if "--m" in parameter_list:
        it_aug = sys.argv[parameter_list.index("--m")+1]
    else:
        it_aug = 3
    if "--max_states" in parameter_list:
        max_states = sys.argv[parameter_list.index("--max_states")+1]
    else:
        max_states = 20
    if "--adv_method" in parameter_list:
        adv_method = sys.argv[parameter_list.index("--adv_method")+1]
    else:
        adv_method = 'H'
    if "--competitor" in parameter_list:
        competitors = sys.argv[parameter_list.index("--competitor")+1]
    else:
        competitors = 'D,O'
    if "--reps" in parameter_list:
        repetitions = sys.argv[parameter_list.index("--reps")+1]
    else:
        repetitions = 1
    if "--ncpus" in parameter_list:
        ncpus = int(sys.argv[parameter_list.index("--ncpus")+1])
    else:
        ncpus = 1
    
    
    if adv_method not in recognized_adv_methods:
        print("Adversarial method in input is not recognized")
        print("For now the list is", recognized_adv_methods)
        sys.exit()
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    train_sizes = train_sizes.strip().split(',')
    competitors = competitors.strip().split(',')
    #print(competitors)
    
    querys = [(path_train, path_test, path_gt, output_dir, t, pca_components, w, eps, it_aug, max_states, adv_method, competitors, repetitions) for t in train_sizes]
    with Pool(processes=ncpus) as pool:
        pool.starmap(train, querys)
    #for train_size in train_sizes:
    #    train(path_train, path_test, path_gt, output_dir, train_size, pca_components, w, eps, it_aug, max_states, adv_method, repetitions)
    
