# Hemodynamic_parameter_estimation_based_on_PCG_and_PPG
Here are the codes and data used in the paper: 
"Continuous Estimation of Left Ventricular Hemodynamic Parameters Based on Heart Sound and PPG Signals Using Deep Neural Network"

The folder named 'data' contains the preprocessed data for two subjects:
For subject 1, 15 records have been  collected.
For subject 2, 14 records are included.

Each record has been split into cardiac cycles in '.mat' format. 
Each '.mat' file contains three variables: 'pcg_ppg', 'label', and 'length'. 
'pcg_ppg' is a vector whose size is 2*1000. The first row of 'pcg_ppg' is padded heart sound signal and the second is padded PPG signal.
The 'label' contains the values of the four  target parameters.  
'length' is the length of the raw data without padding.


There are three scripts:
1.train_model_for_scheme0.py,
2.train_model_for_scheme_I.py,
which are the realization of the corresponding validation schemes introduced in our paper.


Requirements for running the code:
python==3.5
numpy==1.16.2
scipy==1.3.1
keras==2.3.1
tensorflow-gpu==1.15.0
scikit-learn
tqdm
