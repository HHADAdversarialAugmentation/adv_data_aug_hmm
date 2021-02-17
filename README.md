# adv_data_aug_hmm
Adversarial Data Augmentation for Anomaly Detection in Intelligent Autonomous Systems

In order to use this software the user needs to create a conda environment (or alternatively having a Python3.8 valid installation with all the dependencies) with the following command:

	conda env create -f env_setup.yml

Afterwards, the user needs to compile the Cython functions by means of the following command:

	python setup_cython_function.py build_ext --inplace
	
	
How to call the function (example):

	python adv_data_aug_hmm.py --train user/data/dataset_train.csv --test user/data/dataset_test.csv 
		--gt user/data/dataset_ground_truth.csv --window 100 --eps 0.05 --it_aug 3 
		--max_states 10 --adv_method H
		
