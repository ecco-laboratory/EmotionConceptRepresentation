import sys
sys.path.append('/home/yma/EmoTEM/scripts/perceptual')

import pandas as pd
import numpy as np
import torch
import glob
import importlib.util
# Own module imports. Note how model module is not imported, since we'll used the model from the training run
import src.world as world
import src.analyse as analyse
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
import random

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
min_index = 0
max_index = 100000 # '32000'
its_to_process = list(range(min_index, max_index, 1000))+[max_index]

walk_random_seed = 42
random.seed(walk_random_seed)
n_random_walkers = 10
# Make list of all the environments that this model was trained on
envs_dir = '/home/yma/EmoTEM/scripts/perceptual/envs/emofilm/'
envs = list(glob.glob(envs_dir + '11x11_13categories_MDSseed121_14brainmovies.json'))#list(glob.glob(envs_dir + '*.json'))
tem_dir = '/home/yma/EmoTEM/scripts/perceptual/Summaries/'

result_folder = os.path.join('./outputs/TEM_performace/across_movies/MDSseed121')
os.makedirs(result_folder, exist_ok=True)
walk_random_seed_folder = os.path.join(result_folder, f'walkRandomSeed{walk_random_seed}')
os.makedirs(walk_random_seed_folder, exist_ok=True)

embedding_files = [None] * len(envs)
index2embeddings = [{int(k): v for k, v in json.load(open(index2embedding, 'r')).items()} if index2embedding else None
                        for index2embedding in embedding_files] 

zero_shot_results = []
for walker in range(n_random_walkers):
    print(f"Processing walker {walker}")
    for env_to_plot in range(len(envs)):
        movie_name = envs[env_to_plot].split('/')[-1].split('_')[-1].split('.')[0]
        env_MDS_seed = envs[env_to_plot].split('/')[-1].split('_')[2]

        # Set which environments will include shiny objects
        shiny_envs = [False]
        envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not shiny_env for shiny_env in shiny_envs]
        # Set the number of walks to execute in parallel (batch size)
        n_walks = len(shiny_envs)
        # Select environments from the environments included in training
        environments = [world.World(graph, randomise_observations=False, shiny=None) 
                        for env_i, graph in enumerate([envs[env_to_plot]])]
        # Determine the length of each walk
        walk_len = np.median([env.n_locations * 100 for env in environments]).astype(int)
        # And generate walks for each environment
        walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

        # Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
        model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
        for i_step, step in enumerate(model_input):
            model_input[i_step][1] = torch.stack(step[1], dim=0)

        for index in tqdm(its_to_process, desc="Processing Iteration"):
            tem = load_model(date, run, str(index), tem_dir)
            tem.eval()

            with torch.no_grad():
                forward = tem(model_input, prev_iter=None)
            zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=True)

            zero_shot_value = np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100
            zero_shot_results.append({
                'zero_shot': zero_shot_value,
                'iteration': index,
                'TEMwalker': walker
            })
        if len(envs) == 1:
            #save zero_shot_results as a csv for eadh walker iteractively 
            zero_shot_results_df = pd.DataFrame(zero_shot_results)
            
            # Write or append to the CSV file after each walker
            zero_shot_results_df.to_csv(os.path.join(walk_random_seed_folder, f'zero_shot_performance_{movie_name}_env{env_MDS_seed}.csv'),
                                         mode='a', header=(walker == 0), index=False)
            
            # Clear the results for the next walker
            zero_shot_results.clear()

            