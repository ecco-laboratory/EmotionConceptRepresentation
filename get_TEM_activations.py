import sys
sys.path.append('/home/yma/EmoTEM/scripts/perceptual')

from collections import defaultdict
import numpy as np
import torch
import glob
import importlib.util
# Own module imports. Note how model module is not imported, since we'll used the model from the training run
import src.world as world
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
import random
import argparse

def remap_keys(category_dict, mapping):
    new_category_dict = defaultdict(list)
    for old_key, values in category_dict.items():
        new_key = mapping.get(int(old_key))
        if new_key:
            new_category_dict[new_key].extend(values)
    return new_category_dict

def get_pg_activation(forward, model, environments, obs_to_include, n_steps_removed=0, obs_to_category=None, freqs_list = 'all'):
    """ Get the observation*n_cells matrix of p&g activations of the specified frequency modules.
    Args:
        freqs: list of lists of frequency modules to combine
    Returns:

    """
    # Go through environments and collect firing rates in each
    if freqs_list == 'all':
        freqs_list = [list(range(model.hyper['n_f']))]
    g_allobs_allfreqs_allenvs, p_allobs_allfreqs_allenvs = [], []
    for env_i, env in enumerate(environments):
        g_allobs_allfreqs, p_allobs_allfreqs = {}, {}
        for freqs in freqs_list:
            g_allsteps, p_allsteps = defaultdict(list), defaultdict(list)
            for step in forward[n_steps_removed:]:
                true_obs_index = np.argmax(step.pred_true()[1][env_i].numpy())
                if str(true_obs_index) in [str(obs) for obs in obs_to_include]:
                    # Run through frequency modules and append the firing rates to the correct location list
                    g_allsteps[str(true_obs_index)].append(np.concatenate([step.g_inf[f][env_i].numpy() for f in freqs]))
                    p_allsteps[str(true_obs_index)].append(np.concatenate([step.p_inf[f][env_i].numpy() for f in freqs]))
                    
            if obs_to_category is not None:
                g_allsteps = remap_keys(g_allsteps, obs_to_category)
                p_allsteps = remap_keys(p_allsteps, obs_to_category)
            g_allobs, p_allobs = defaultdict(list), defaultdict(list)
            for key_g, arr_g_list in g_allsteps.items():
                if isinstance(arr_g_list, list):
                    g_allobs[key_g] = np.mean(arr_g_list, axis=0).tolist()  # Average across visits
            for key_p, arr_p_list in p_allsteps.items():
                if isinstance(arr_p_list, list):
                    p_allobs[key_p] = np.mean(arr_p_list, axis=0).tolist()
            g_allobs_allfreqs['freq'+"".join([str(f) for f in freqs])] = g_allobs
            p_allobs_allfreqs['freq'+"".join([str(f) for f in freqs])] = p_allobs
        g_allobs_allfreqs_allenvs.append(g_allobs_allfreqs)
        p_allobs_allfreqs_allenvs.append(p_allobs_allfreqs)
    return g_allobs_allfreqs_allenvs, p_allobs_allfreqs_allenvs
             

def load_model(date, run, index, tem_dir):
    # Load the model: use import library to import module from specified path
    model_spec = importlib.util.spec_from_file_location("model", tem_dir + date + '/run' + run + '/script/src/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)
    # Load the parameters of the model
    params = torch.load(tem_dir + date + '/run' + run + '/model/params_' + index + '.pt', map_location=torch.device('cpu'))
    new_params = {'device':'cpu'}
    for key in new_params:
        params[key] = new_params[key]

    # Create a new tem model with the loaded parameters
    tem = model.Model(params)
    # Load the model weights after training
    model_weights = torch.load(tem_dir + date + '/run' + run + '/model/tem_' + index + '.pt', map_location=torch.device('cpu'))
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    return tem

date =  '2024-10-22' # '2020-10-13' 
run = '0' # '0'
##min_index = 200000
#max_index = 200000 # '32000'
#its_to_process = list(range(min_index, max_index, 1000))+[max_index]
its_to_process = [48000]#[40000, 50000, 32000]#[32000]
parser = argparse.ArgumentParser(description="Get TEM activations with a specified random seed.")
parser.add_argument(
    "--seed",
    type=int,
    default=42,  # Default seed value
    help="Random seed for walk generation (default: 42)"
)
walk_random_seed = parser.parse_args().seed
random.seed(walk_random_seed)
n_random_walkers = 29
# Make list of all the environments that this model was trained on
envs_dir = '/home/yma/EmoTEM/scripts/perceptual/envs/emofilm/'
envs = list(glob.glob(envs_dir + '11x11_13categories_MDSseed468_14brainmovies.json'))#list(glob.glob(envs_dir + '*.json'))
tem_dir = '/home/yma/EmoTEM/scripts/perceptual/Summaries/'

result_folder = os.path.join('.', 'outputs', 'TEM_activation', 'across_movies', 'MDSseed468')
os.makedirs(result_folder, exist_ok=True)

for walker in range(n_random_walkers):
    for env_to_plot in range(len(envs)):
        movie_name = envs[env_to_plot].split('/')[-1].split('_')[-1].split('.')[0]
        env_MDS_seed = envs[env_to_plot].split('/')[-1].split('_')[2]
        print(f"Processing {movie_name}")
        env_file = open(envs[env_to_plot], 'r')
        env_json = json.load(env_file)
        emo_obs = [location['observation'] for location in env_json['locations'] if location['category'] != 'empty']
        all_obs = [i for i in range(env_json['n_locations'])]
        obs_to_include = emo_obs

        obs_to_category = {env_json['locations'][i]['observation']: env_json['locations'][i]['category'] for i in range(env_json['n_locations']) if env_json['locations'][i]['observation'] in obs_to_include}
        # Set which environments will include shiny objects
        shiny_envs = [False]
        # Set the number of walks to execute in parallel (batch size)
        n_walks = len(shiny_envs)
        # Select environments from the environments included in training
        environments = [world.World(graph, randomise_observations=False, shiny=None) 
                        for env_i, graph in enumerate([envs[env_to_plot]])]
        # Determine the length of each walk
        walk_len = np.median([env.n_locations * 100 for env in environments]).astype(int)
        n_steps_removed = int(walk_len/2)#env_json['n_locations'] #remove the first n_locations steps from analysis for TEM to see all nodes to stabilize
        # And generate walks for each environment
        walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

        # Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
        model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
        for i_step, step in enumerate(model_input):
            model_input[i_step][1] = torch.stack(step[1], dim=0)

        for index in tqdm(its_to_process, desc="Processing Iteration"):
            tem = load_model(date, run, str(index), tem_dir)
            tem.eval()
            print(f'Number of frequencies: {tem.hyper["n_f"]}')
            freqs_list = [[f] for f in list(range(tem.hyper['n_f']))]
            with torch.no_grad():
                forward = tem(model_input, prev_iter=None)
            g_activation, p_activation = get_pg_activation(forward, tem, environments, obs_to_include, n_steps_removed, obs_to_category, freqs_list)

            iteration_folder = os.path.join(result_folder, f'iteration{index}')
            os.makedirs(iteration_folder, exist_ok=True)
            walk_random_seed_folder = os.path.join(iteration_folder, f'walkRandomSeed{walk_random_seed}')
            os.makedirs(walk_random_seed_folder, exist_ok=True)

            p_activation = p_activation[env_to_plot]
            with open(os.path.join(walk_random_seed_folder, f'p_activation_{movie_name}_env{env_MDS_seed}_randomWalker{walker}.json'), 'w') as f:
                json.dump(p_activation, f)

            g_activation = g_activation[env_to_plot]
            with open(os.path.join(walk_random_seed_folder, f'g_activation_{movie_name}_env{env_MDS_seed}_randomWalker{walker}.json'), 'w') as f:
                json.dump(g_activation, f)
        
        
        