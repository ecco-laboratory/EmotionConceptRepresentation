# The representation of emotion knowledge in hippocampal-prefrontal systems
It is well established that humans conceptualize emotion concepts in a low-dimensional space that resembles a map. For the most part, this map is organized using two-dimensions that reflect the pleasantness and arousal experienced during events that are typically described using specific emotion terms (fear, anger, surprise, and so forth). Recently, [we have hypothesized](https://www.sciencedirect.com/science/article/pii/S0149763425000892) that this map-like structure is present in communication and reasoning about emotions results from  processing in hippocampal-prefrontal systems that bind sensory observations to abstract locations in a relational graph. In binding sensory observations (e.g., the sensations that occur while nearly stepping on a snake) to abstracts locations in a structed graph that could be used for multiple purposes (e.g., mapping relationships between physical space, different objects, or in this case emotion concepts) these systems organize emotion knowledge into a compact, two-dimensional representation that can be used to make predictions about emotional experiences. 

In this project we tested several predictions of this hypothesis using a computational model of relational memory (the Tolman Eichenbaum Machine; [TEM](https://www.sciencedirect.com/science/article/pii/S009286742031388X)) and fMRI data collected as participants viewed a series of [emotional films](https://www.nature.com/articles/s41597-025-04803-5). Probing the information content of the hippocampus, entorhinal cortex, and prefontal cortex, we use multivariate decoding to predict normative [emotion self-report](https://openneuro.org/datasets/ds004872/versions/1.0.3) and representations used by TEM to learn the structure of the environment. A more detailed description of this work will be made available as a preprint shortly. This repository provides the source code used to conduct these analyses.


![alt text](https://github.com/EmotionConceptRepresentation/tree/master/images/EmotionConceptMapping.png?raw=true)



## Dependencies 
This code uses [Canlab Core Tools](https://github.com/canlab/CanlabCore/tree/master), which is an object oriented toolbox that uses the [SPM software](https://www.fil.ion.ucl.ac.uk/spm/) to process fMRI data. Instructions for installing CANlab Core Tools which provided many of the functions and Neuroimaging Pattern Masks used in the analyses can be found [here](https://canlab.github.io/_pages/canlab_help_1_installing_tools/canlab_help_1_installing_tools.html).

Instructions for installing SPM12 can be found [here](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/).

The MATLAB interface used for creating TEM environments is available [here](https://github.com/jbakermans/WorldBuilder).

Code for training TEM is adapted from [here](https://github.com/jbakermans/torch_tem) (python >= 3.6.0 and pytorch >= 1.6.0 required).

The code requires both [PyTorch](https://pytorch.org/) and [Tensorboard](https://www.tensorflow.org/tensorboard) to run. 

Installation should take approximately 5 minutes to run.

## Acquiring the fMRI Dataset
The Emo-FilM fMRI data ([Morgenroth et al., 2025](https://www.nature.com/articles/s41597-025-04803-5)) can be downloaded by running
<pre><code>aws s3 sync --no-sign-request s3://openneuro.org/ds004892/derivatives/preprocessing/ path_to_project_folder/data/fmri</code></pre>

## Example Usage

# Training TEM
Once you have the dependencies installed on your machine, to train the TEM model, navigate to the following directory:
<pre><code>cd TEMcode/</code></pre> 

Then run the following command:
<pre><code>python run.py --env_file ./envs/11x11_square_obs45_act5.json --n_train_iterations 100 --load_existing_model False</code></pre>

Running this script to train the model should take approximately 10 minutes to start generating output.

# fMRI Decoding

Once you have dependencies on your MATLAB path, and downloaded the fMRI data to the ./data/ directory, you can run the following command to perform decoding analyses.

<pre><code>matlab "run('fit_category_decoding_models_brain_generalize_across_movies.m')</code></pre>

or to predict activation in TEM agents

<pre><code>matlab "run('fit_category_decoding_models_brain_generalize_across_movies.m')</code></pre>
