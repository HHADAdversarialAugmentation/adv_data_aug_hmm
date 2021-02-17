# adv_data_aug_hmm
Castellini A., Masillo F., Azzalini D., Amigoni F., Farinelli A., Adversarial Data Augmentation for Anomaly Detection in Intelligent Autonomous Systems

In order to use this software the user needs to create a conda environment (or alternatively having a Python3.8 valid installation with all the dependencies) with the following command:

	conda env create -f env_setup.yml

Afterwards, the user needs to compile the Cython functions by means of the following command:

	python setup_cython_function.py build_ext --inplace


How to call the function (example):

	python adv_data_aug_hmm.py --train data/dataset_train.csv --test data/dataset_test.csv 
		--gt data/dataset_ground_truth.csv --output_dir my_result --train_sizes 250,500,750 
		--pca 3 --w 100 --eps 0.05 --m 3 --max_states 10 --adv_method H --reps 10

PARAMETERS:

	--train : path where train set is
	--test : path where test set is
	--gt : path where ground truth is
	--output_dir : path where the user expects the output
	--train_sizes : comma-separated list of lengths for slicing the train set into subsets
	--pca [OPTIONAL] : number of PCA components to use, this is used to preprocess both train and test set. 
					   If the data is already preprocessed either do not specify it or give 0 as input (DEFAULT=0)
	--w [OPTIONAL] : window size (DEFAULT=100)
	--eps [OPTIONAL] : distance boundary for the generation of adversarial examples (DEFAULT=0.05)
	--m [OPTIONAL] : number of data augmentation steps (DEFAULT=3)
	--max_states [OPTIONAL] : maximum number of states for the HMM, used to choose the optimal number of states based 
							  on the evaluation of BIC (DEFAULT=2)
	--adv_method [OPTIONAL] : adversarial generation algorithm to use. For now we have available 'H' (Hellinger Distance based)
							  and 'L' (Likelihood based) methods (DEFAULT=H)
	--reps [OPTIONAL] : number of repetitions of the whole process. If the train size specified is less than the total length 
						of the train set then this is useful to see how starting from different time instants (taking a slice
						from start to start+train_sizes of the data) can impact both on original and augmented performaces
						(DEFAULT=1)


OUTPUT:
In the output directory the user should expect the following results:

	- a csv file for each train size storing F1 scores for before and after augmentation (e.g. f1_scores_train_size_500_adv_method_H.csv).
	  This file has 2 columns (F1 before and F1 after) and #repetitions rows.
	- a csv file for each train size storing thresholds before and after augmentation (e.g. taus_train_train_size_500_adv_method_H.csv).
	  This file has 2 columns (Tau before and Tau after) and #repetitions rows.
	- a folder for each train size called models_train_size containing pickle files two for each repetition, i.e. the baseline model
	  (e.g. model_1_HHAD.pkl) and the augmented model (e.g. model_1_H-AUG.pkl)
	

If the user wishes to visualize a simple plot of F1 scores across train sizes for a given result, we make available a simple script called plot_f1.py. It can be called as following:

	python plot_f1.py --result_dir my_result --adv_method H --train_sizes 250,500,750
	
It will output a plot with two lines for the mean F1 score (across repetitions for a given train size), one for the baseline model and one for the augmented one.