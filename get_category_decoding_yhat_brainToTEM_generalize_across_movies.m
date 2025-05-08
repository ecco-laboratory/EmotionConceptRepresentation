clear all; close all; clc;
%% Define relevant script-specific constants
folder_project = fileparts(mfilename('fullpath'));
addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks/'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')

stim_names = {
'After_The_Rain', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Damaged_Kung_Fu', ...
'First_Bite', 'Lesson_Learned', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', ...
'Superhero', 'Tears_of_Steel', 'The_secret_number', 'To_Claire_From_Sonny', 'You_Again'
};
% FOR DEBUG/TEST ONLY. DELETE WHEN THIS IS A FUNCTION ARG
bids_task_names = {
'AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'DamagedKungFu', ...
'FirstBite', 'LessonLearned', 'Payload', 'RidingTheRails', 'Sintel', 'Spaceman', ...
'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain'
};

%beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
beh_data = load('/home/data/eccolab/VisionLanguageEncodingEmotion/BehavioralRatingsPerVideoAndDim.mat')
behTab = beh_data.behTab;
% List of categories to retain
emotions_category = {'Anger', 'Anxiety', 'Fear', 'Surprise', 'Guilt', 'Disgust', ...
'Sad', 'Regard', 'Satisfaction', 'WarmHeartedness', 'Happiness', ...
'Pride', 'Love'};
% Subset ratings to only include emotion categories
behTab_category = structfun(@(tbl) tbl(:, emotions_category(ismember(emotions_category, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
%behTab_category = shuffleTableCategories(behTab_category);
label_names = behTab_category.(bids_task_names{1}).Properties.VariableNames;
%ensure category names are in the same order for all movies
for t = 2:length(bids_task_names)
    assert(isequal(label_names, behTab_category.(bids_task_names{t}).Properties.VariableNames), 'Category names are not the same for all movies');
end
pls_n_components = 20;

used_movies = find(~ismember(bids_task_names, {'DamagedKungFu', 'RidingTheRails'}));
num_movies = length(used_movies);

%% Specify paths and constants for TEM
walk_random_seed = 42;
num_walkers = 10;

TEM_iterations = {32000, 42000, 50000};% 50000

walkers = arrayfun(@(x) ['randomWalker' num2str(x)], 0:(num_walkers-1), 'UniformOutput', false);
%{
component_names = {'p', 'g'};
frequencies_allcomponents = {{{'freq0', 'freq1', 'freq2', 'freq3', 'freq4'}},... %for p
                {{'freq0', 'freq1', 'freq2', 'freq3', 'freq4'}}... %for g
                };
frequency_names_allcomponents = {{'freq01234'},...%for p
                   {'freq01234'}...... %for g
                   };
%}

                   
%{
component_names = {'p'};
frequencies_allcomponents = {{{'freq0', 'freq1'}, {'freq2', 'freq3', 'freq4'}}... %for p
                            };
frequency_names_allcomponents = {{'freq01', 'freq234'}...%for p
                                };
%}
component_names = {'g'};
frequencies_allcomponents = {{{'freq4'}, {'freq3', 'freq4'}, {'freq2', 'freq3', 'freq4'}}... %for g
                            };
frequency_names_allcomponents = {{'freq4', 'freq34', 'freq234'}...%for g
                                };

%% Specify paths and constants for brain
% The string after subject, session, task, and other changing items in the BIDS-compatible filename
bold_suffix = 'space-MNI_desc-ppres_bold.nii';
% STUDY-SPECIFIC! SHOULD BE AN ARG
tr_length = 1.3;

brain_atlas = load_atlas('canlab2018');
region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz'),...
                select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'})};
region_names = {'Hippocampus', 'EntorhinalCortex', 'vmPFC'};

%{
region_masks = {fullfile(folder_project, 'masks', 'HC_ant_Julich.nii'),...
                fullfile(folder_project, 'masks', 'HC_post_Julich.nii')};
region_names = {'anteriorHippocampus', 'posteriorHippocampus'};
%}

% Brain folder should go directly to the folder containing the subject subfolders
folder_brain_5subs = '/home/data/eccolab/VisionLanguageEncodingEmotion/Code/EmotionConcepts/data/aws/';
subjects_5 = {'sub-S22', 'sub-S25', 'sub-S26', 'sub-S29', 'sub-S32'};
folder_brain_24subs = '/home/data/eccolab/OpenNeuro/ds004892/derivatives/preprocessing/';
subjects_24 = {dir(fullfile(folder_brain_24subs, 'sub-*')).name};
%subjects_24 = setdiff(subjects_24, [subjects_5, 'sub-S07', 'sub-S01']);  
subjects_24 = setdiff(subjects_24, [subjects_5, 'sub-S07']);%drop S07 and the 5 subjects from the 5 subject set
subjects = [subjects_24, subjects_5];
sessions = {dir(fullfile(folder_brain_24subs, subjects{1}, 'ses-*')).name};


selected_regions = 1:length(region_masks);
num_regions = length(selected_regions);
num_subjects = length(subjects);

%% Main loop
for s = 1:num_subjects
    if ismember(subjects{s}, subjects_5)
        folder_brain = folder_brain_5subs;
    else
        folder_brain = folder_brain_24subs;
    end
    try 
        %% Get nifti file paths for all movies
        files_all_movies = {};
        n_trs = {};
        available_movies = {};
        for t = used_movies
            task = bids_task_names{t};
            for session = sessions
                file = [folder_brain subjects{s}  '/' session{1} '/func/'  subjects{s} '_' session{1} '_task-' task '_' bold_suffix];
                if exist(file, 'file')
                    fprintf('Found file %s\n', file);
                    files_all_movies = [files_all_movies; file];
                    n_trs = [n_trs, length(spm_vol(file))];
                    available_movies = [available_movies, t];
                    break
                end
            end
            if length(available_movies) > 1
                current_movie_index = available_movies(end);
                last_movie_index = available_movies(end-1);
                if current_movie_index{1} ~= last_movie_index{1}+1
                    n_trs = [n_trs(1:end-1), {0}, n_trs(end)]; % Add a zero for tr of missing movie
                end
            end
        end
        %% Load the fMRI data together for all movies
        fprintf('Loading fmri data from all movies for subject %s\n', subjects{s})
        dat_all_movies = fmri_data(files_all_movies);
        masked_dat_all_movies = struct();
        for r = selected_regions
            % Apply the mask
            mask = region_masks{r}; %select_atlas_subset(load_atlas('canlab2018'), regions{r});
            masked_dat_all_movies.(region_names{r}) = apply_mask(dat_all_movies, mask);
        end
        clear dat_all_movies; % Clear the original data to save memory

        %% Get concatenated BOLD data across movies for all ROIs
        n_trs = [1, n_trs];
        t_id = 1;
        kinds = [];
        all_regions_concat_bold = struct();
        fprintf('Concatenating BOLD data for subject %s\n', subjects{s})
        for t = available_movies
            t=t{1};
            try
                % Load the behavioral ratings and resample to match BOLD TR
                normative_self_report = table2array(behTab_category.(bids_task_names{t}));
                normative_self_report = fillmissing(normative_self_report, 'nearest', 1);
                normative_self_report = resample(double(normative_self_report), 10, 13);  % Resample for BOLD TR

                kinds = [kinds; t_id * ones(height(normative_self_report), 1)];
                t_id = t_id + 1;
                
                
                for r = selected_regions
                    masked_dat_current_movie = masked_dat_all_movies.(region_names{r}).dat(:, sum(cell2mat(n_trs(1:t))):(sum(cell2mat(n_trs(1:t+1))) - 1));
                    
                    starting_tr = round(90 / tr_length);
                    masked_dat_current_movie = masked_dat_current_movie(:, starting_tr + (1:height(normative_self_report)));
                    masked_dat_current_movie = masked_dat_current_movie';  % Transpose to time x voxels

                    if ~isfield(all_regions_concat_bold, region_names{r})
                        all_regions_concat_bold.(region_names{r}) = []; % Initialize the field as an empty array
                    end
                    all_regions_concat_bold.(region_names{r}) = [all_regions_concat_bold.(region_names{r}); masked_dat_current_movie];
                end
            catch ME
                fprintf('Error processing subject %s, movie %s: %s\n', subjects{s}, bids_task_names{t}, ME.message);
            end
        end

        for iter = TEM_iterations
            TEM_iteration = iter{1};
            % Output directory
            output_dir = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_yhat', 'brainToTEM', 'MDSseed121');
            output_dir = fullfile(output_dir, sprintf('iteration%d', TEM_iteration), sprintf('walkRandomSeed%d', walk_random_seed));
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            folder_tem_activation = fullfile(folder_project, 'outputs', 'TEM_activation', 'across_movies', 'MDSseed121');
            folder_tem_activation = fullfile(folder_tem_activation, sprintf('iteration%d', TEM_iteration), sprintf('walkRandomSeed%d', walk_random_seed));
            tem_activation_jsons = dir(fullfile(folder_tem_activation, '*.json'));
            
            %% PLS looping over all TEM walkers and components
            for component = 1:length(component_names)
                y_table_struct = struct();
                yhat_table_struct = struct();
                for tem_s = walkers
                    component_name = component_names{component};
                    frequencies = frequencies_allcomponents{component};
                    frequency_names = frequency_names_allcomponents{component};
                    num_freqs = length(frequencies);
                    %% Get concatenated TEM activation data across movies for all frequency groups
                    concat_pg_activation_ts_allFreqGroups = struct();
                    cell_names_allFreqGroups = struct();
                    pg_json = fullfile(folder_tem_activation, [component_name '_activation_14brainmovies_envMDSseed121_' tem_s{1} '.json']);
                    pg_json_data = jsondecode(fileread(pg_json));
                    fprintf('Concatenating TEM activation data for walker %s and component %s\n', tem_s{1}, component_name);
                    for f_group = 1:num_freqs
                        pg_activation_ts_all_movies = [];
                        for t = available_movies
                            t=t{1};
                            normative_self_report = table2array(behTab_category.(bids_task_names{t}));
                            normative_self_report = fillmissing(normative_self_report, 'nearest', 1);
                            pg_activation_ts_current_movie = generate_pg_activation_timeseries(pg_json_data, frequencies{f_group}, normative_self_report, behTab_category.(bids_task_names{1}).Properties.VariableNames);
                            pg_activation_ts_current_movie = resample(double(pg_activation_ts_current_movie), 10, 13);
                            % Convolve with HRF to match BOLD signal
                            for i = 1:size(pg_activation_ts_current_movie, 2)
                                tmp = conv(double(pg_activation_ts_current_movie(:, i)), spm_hrf(tr_length));
                                conv_pg(:, i) = tmp(:); 
                                clear tmp;
                            end
                            % Trim convolved timeseries to match original length
                            conv_pg = conv_pg(1:height(pg_activation_ts_current_movie), :);
                            pg_activation_ts_all_movies = [pg_activation_ts_all_movies; conv_pg];
                            clear conv_pg;
                        end
                        concat_pg_activation_ts_allFreqGroups.(frequency_names{f_group}) = pg_activation_ts_all_movies;
                        % Get cell names (frequency+cell index) for column names of the performance table
                        cell_names_allFreqGroups.(frequency_names{f_group}) = [];
                        for freq = frequencies{f_group}
                            temp_field = fieldnames(pg_json_data.(freq{1}));
                            num_cells = size(pg_json_data.(freq{1}).(temp_field{1}), 1);
                            cell_names = arrayfun(@(idx) sprintf('%scell%d', freq{1}, idx), 0:num_cells-1, 'UniformOutput', false);
                            cell_names_allFreqGroups.(frequency_names{f_group}) = [cell_names_allFreqGroups.(frequency_names{f_group}), cell_names];
                        end
                    end
                    y_table_struct.(tem_s{1}) = struct();
                    yhat_table_struct.(tem_s{1}) = struct();
                    %% Run PLS regression and compute performance
                    for r = selected_regions
                        y_table_struct.(tem_s{1}).(region_names{r}) = struct();
                        yhat_table_struct.(tem_s{1}).(region_names{r}) = struct();
                        concat_bold = all_regions_concat_bold.(region_names{r});
                        for f_group = 1:num_freqs
                            fprintf('Running PLS regression for region %s, walker %s, component %s, and frequency %s\n', region_names{r}, tem_s{1}, component_name, frequency_names{f_group});
                            concat_pg = concat_pg_activation_ts_allFreqGroups.(frequency_names{f_group});
                            % Run PLS regression and compute performance
                            [~, ~, ~, ~, b] = plsregress(concat_bold, concat_pg, pls_n_components);
                            clear yhat pred* diag* pred_obs_corr* diag_corr*;
                            movie_column = {};
                            for k = 1:max(kinds)
                                [~, ~, ~, ~, beta_cv] = plsregress(concat_bold(kinds ~= k, :), concat_pg(kinds ~= k, :), pls_n_components);
                                yhat(kinds == k, :) = [ones(length(find(kinds == k)), 1) concat_bold(kinds == k, :)] * beta_cv;
                                current_available_movie_name = string(bids_task_names{available_movies{k}});
                                movie_column = [movie_column; repmat({current_available_movie_name}, length(find(kinds == k)), 1)];
                            end
                            yhat_table = array2table(yhat, 'VariableNames', cell_names_allFreqGroups.(frequency_names{f_group}));
                            yhat_table.movie = movie_column;
                            y_table = array2table(concat_pg, 'VariableNames', cell_names_allFreqGroups.(frequency_names{f_group}));
                            y_table.movie = movie_column;
                            y_table_struct.(tem_s{1}).(region_names{r}).(frequency_names{f_group}) = y_table;
                            yhat_table_struct.(tem_s{1}).(region_names{r}).(frequency_names{f_group}) = yhat_table;
                        end 
                    end
                end
                %average across tem_s
                for r = selected_regions
                    for f_group = 1:num_freqs
                        walker_names = fieldnames(y_table_struct);
                        y_table = y_table_struct.(walker_names{1}).(region_names{r}).(frequency_names{f_group});
                        yhat_table = yhat_table_struct.(walker_names{1}).(region_names{r}).(frequency_names{f_group});
                        movie_column = y_table.movie;
                        %remove movie column
                        y_table = removevars(y_table, 'movie');
                        yhat_table = removevars(yhat_table, 'movie');
                        y_reference_var_names = y_table.Properties.VariableNames;
                        yhat_reference_var_names = yhat_table.Properties.VariableNames;
                        for tem_s = walker_names(2:end)
                            current_y_table = y_table_struct.(tem_s{1}).(region_names{r}).(frequency_names{f_group});
                            current_yhat_table = yhat_table_struct.(tem_s{1}).(region_names{r}).(frequency_names{f_group});
                            current_y_table = removevars(current_y_table, 'movie');
                            current_yhat_table = removevars(current_yhat_table, 'movie');
                            %ensure the variable names are the same, if not, reorder them to match the reference
                            current_y_var_names = current_y_table.Properties.VariableNames;
                            current_yhat_var_names = current_yhat_table.Properties.VariableNames;
                            if ~isequal(y_reference_var_names, current_y_var_names)
                                [~, idx] = ismember(y_reference_var_names, current_y_var_names);
                                if any(idx == 0) || isempty(idx)
                                    error('Variable names in y_table do not match the reference variable names');
                                end
                                current_y_table = current_y_table(:, idx);
                            end
                            if ~isequal(yhat_reference_var_names, current_yhat_var_names)
                                [~, idx] = ismember(yhat_reference_var_names, current_yhat_var_names);
                                if any(idx == 0) || isempty(idx)
                                    error('Variable names in yhat_table do not match the reference variable names');
                                end
                                current_yhat_table = current_yhat_table(:, idx);
                            end
                            y_table = table2array(y_table) + table2array(current_y_table);
                            yhat_table = table2array(yhat_table) + table2array(current_yhat_table);
                        end
                        y_table = y_table / num_walkers;
                        yhat_table = yhat_table / num_walkers;
                        y_table = array2table(y_table, 'VariableNames', y_reference_var_names);
                        yhat_table = array2table(yhat_table, 'VariableNames', yhat_reference_var_names);
                        y_table.movie = movie_column;
                        yhat_table.movie = movie_column;
                        y_output_file = fullfile(output_dir, sprintf('y_%s_%s_%s_%s.csv',  subjects{s}, region_names{r}, component_name, frequency_names{f_group}));
                        yhat_output_file = fullfile(output_dir, sprintf('yhat_%s_%s_%s_%s.csv',  subjects{s}, region_names{r}, component_name, frequency_names{f_group}));
                        writetable(y_table, y_output_file);
                        writetable(yhat_table, yhat_output_file);
                        fprintf('Wrote y and yhat tables to %s and %s\n', y_output_file, yhat_output_file);
                    end
                end
                clear y_table_struct yhat_table_struct;
            end
        end
    catch ME
        fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end



function pg_activation_ts = generate_pg_activation_timeseries(pg_json_data, freq_group, category_rating_ts, category_names)
    [num_timepoints, num_categories] = size(category_rating_ts);
    if length(category_names) ~= num_categories
        error('The number of category names must match the number of columns in category_rating_ts.');
    end
    all_activation_data = [];
    % Iterate over the specified frequencies in freq_group
    for freq_idx = 1:length(freq_group)
        freq_key = freq_group{freq_idx};
        if ~isfield(pg_json_data, freq_key)
            error('Frequency key "%s" not found in the JSON data.', freq_key);
        end
        % Get the activation data for all categories in the current frequency
        freq_data = pg_json_data.(freq_key);
        % Initialize a cell array to hold activation data for this frequency
        freq_cat_activation = [];
        for cat_idx = 1:num_categories
            category_name = category_names{cat_idx};
            if ~isfield(freq_data, category_name)
                error('Category "%s" not found in frequency "%s".', category_name, freq_key);
            end

            % Get the activation data for the current category
            cat_activation = freq_data.(category_name);

            % Ensure the activation data is a column vector (num_cells x 1)
            if size(cat_activation, 2) ~= 1
                error('Activation data for category "%s" in frequency "%s" must be a column vector.', category_name, freq_key);
            end

            % Concatenate activations for all categories horizontally
            freq_cat_activation = [freq_cat_activation, cat_activation];
        end

        % Store the horizontally concatenated activations for this frequency
        all_activation_data = [all_activation_data; freq_cat_activation];
    end

    % Get the total number of cells across all frequencies
    num_cells = size(all_activation_data, 1);

    % Initialize the output timeseries matrix
    pg_activation_ts= zeros(num_timepoints, num_cells);

    % Compute the weighted average for each time point
    for t = 1:num_timepoints
        % Get the weights for each category at time t
        weights = category_rating_ts(t, :);

        % Normalize the weights to sum to 1
        weights = weights / sum(weights);

        % Compute the weighted average activation for this time point
        weighted_activation = zeros(num_cells, 1);

        for cat_idx = 1:num_categories
            category_weight = weights(cat_idx);

            % Get the activation data for the current category
            category_activation = all_activation_data(:, cat_idx);

            % Apply the weighting
            weighted_activation = weighted_activation + category_weight * category_activation;
        end
        pg_activation_ts(t, :) = weighted_activation';
    end
end