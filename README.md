# ADMM-ESINet

1. The Python code for ADMM ESINet is as follows: You need to launch Brainstorm and execute data_generator_simulation_300 (MATLAB/Data Generate) to generate EEG data for the training set, while modifying the path parameters in the network. Run the 21main_trans function to start training the model.

2. Dataset Provision: The Brainstorm Epilepsy dataset can be accessed at https://neuroimage.usc.edu/brainstorm/DatasetEpilepsy. The Yokogawa dataset can be obtained from https://neuroimage.usc.edu/brainstorm/Tutorials/Yokogawa, and the Localize-mi dataset can be found at https://doi.org/10.25493/NXN2-05W. We have ensured that all data processing steps comply with the guidelines outlined on the respective websites.
The high-resolution source imaging network has been placed under the /High-resolution experimental network directory. Source imaging results are provided for different numbers of brain sources, including cases with 4006, 5006, and 6006 sources.

3. The imaging results for other subjects in the Localize-MI dataset are included in the Supplement Document.


Due to the potential differences in the amplitudes of the generated simulated data, $\lambda$ needs to be adjusted for simulated data with different amplitudes (this parameter influences the range of the subsequent sparsity threshold).
