# The representation of emotion knowledge in hippocampal-prefrontal systems


## Dependencies 
Instructions for installing CANlab Core Tools which provided many of the functions and Neuroimaging Pattern Masks used in the analyses can be found [here](https://canlab.github.io/_pages/canlab_help_1_installing_tools/canlab_help_1_installing_tools.html).

Instructions for installing SPM12 can be found [here](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/).

The MATLAB interface used for creating TEM environments is available [here](https://github.com/jbakermans/WorldBuilder).

Code for training TEM model is adapted from [here](https://github.com/jbakermans/torch_tem) (python >= 3.6.0 and pytorch >= 1.6.0 required).
To train the TEM model, navigate to the following directory:
<pre><code>cd TEMcode/2024-10-22/run0/script/</code></pre>
Then run the following command:
<pre><code>python run.py --env_file ./envs/11x11_square_obs45_act5.json --n_train_iterations 1000000 --load_existing_model False</code></pre>

## Dataset
The Emo-FilM fMRI data ([Morgenroth et al., 2025](https://www.nature.com/articles/s41597-025-04803-5)) can be downloaded by running
<pre><code>aws s3 sync --no-sign-request s3://openneuro.org/ds004892/derivatives/preprocessing/ path_to_project_folder/data/fmri</code></pre>
