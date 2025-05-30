set_up_paths_and_data_directories;

stim_names = {
'After_The_Rain', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Damaged_Kung_Fu', ...
'First_Bite', 'Lesson_Learned', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', ...
'Superhero', 'Tears_of_Steel', 'The_secret_number', 'To_Claire_From_Sonny', 'You_Again'
};
bids_task_names = {
'AfterTheRain', 'BetweenViewings', 'BigBuckBunny', 'Chatter', 'DamagedKungFu', ...
'FirstBite', 'LessonLearned', 'Payload', 'RidingTheRails', 'Sintel', 'Spaceman', ...
'Superhero', 'TearsOfSteel', 'TheSecretNumber', 'ToClaireFromSonny', 'YouAgain'
};

beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
behTab = beh_data.behTab;
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

TEM_iterations = {32000}; %{40000, 32000, 50000};% 50000

walkers = arrayfun(@(x) ['randomWalker' num2str(x)], 0:(num_walkers-1), 'UniformOutput', false);

component_names = {'p', 'g'};
frequencies_allcomponents = {{{'freq0', 'freq1', 'freq2', 'freq3', 'freq4'}},... %for p
                {{'freq0', 'freq1', 'freq2', 'freq3', 'freq4'}}... %for g
                };
frequency_names_allcomponents = {{'freq01234'},...%for p
                   {'freq01234'}...... %for g
                   };

%% Specify paths and constants for brain
bold_suffix = 'space-MNI_desc-ppres_bold.nii';
tr_length = 1.3;

brain_atlas = load_atlas('canlab2018');

region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz'),...
                select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'})};
region_names = {'Hippocampus', 'EntorhinalCortex', 'vmPFC'};

% Brain folder should go directly to the folder containing the subject subfolders 
subjects = {dir(fullfile(folder_brain, 'sub-*')).name};
subjects = setdiff(subjects, 'sub-S07');%drop S07 
sessions = {dir(fullfile(folder_brain, subjects{1}, 'ses-*')).name};


selected_regions = 1:length(region_masks);
num_regions = length(selected_regions);
num_subjects = length(subjects);

tv = struct(); 
subjnames = struct();
for s = 1:num_subjects
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
                        all_regions_concat_bold.(region_names{r}) = []; 
                    end
                    all_regions_concat_bold.(region_names{r}) = [all_regions_concat_bold.(region_names{r}); masked_dat_current_movie];
                end
            catch ME
                fprintf('Error processing subject %s, movie %s: %s\n', subjects{s}, bids_task_names{t}, ME.message);
            end
        end

        for iter = TEM_iterations
            TEM_iteration = iter{1};

            output_dir = fullfile(folder_project, 'outputs', 'brain_weight_maps', 'PLSbeta', 'brainToTEM', 'MDSseed121');
            output_dir = fullfile(output_dir, sprintf('iteration%d', TEM_iteration), sprintf('walkRandomSeed%d', walk_random_seed));
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            folder_tem_activation = fullfile(folder_project, 'outputs', 'TEM_activation', 'across_movies', 'MDSseed121');
            folder_tem_activation = fullfile(folder_tem_activation, sprintf('iteration%d', TEM_iteration), sprintf('walkRandomSeed%d', walk_random_seed));
            tem_activation_jsons = dir(fullfile(folder_tem_activation, '*.json'));
            
            %% PLS looping over all TEM walkers and components
            tem_s_idx = 0; 
            subj_avg = struct();
            for tem_s = walkers
                tem_s_idx = tem_s_idx + 1;
                for component = 1:length(component_names)
                    component_name = component_names{component};
                    frequencies = frequencies_allcomponents{component};
                    frequency_names = frequency_names_allcomponents{component};
                    num_freqs = length(frequencies);
                    %% Get concatenated TEM activation data across movies for all frequency groups
                    concat_pg_activation_ts_allFreqGroups = struct();
                    num_cells_per_freq_allFreqGroups = struct();
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

                        num_cells_per_freq = cellfun(@(freq) size(pg_json_data.(freq), 1), frequencies{f_group});
                        num_cells_per_freq_allFreqGroups.(frequency_names{f_group}) = num_cells_per_freq;
                    end
                    %% Run PLS regression and compute performance
                    for r = selected_regions
                        concat_bold = all_regions_concat_bold.(region_names{r});
                        for f_group = 1:num_freqs
                            fprintf('Running PLS regression for region %s, walker %s, component %s, and frequency %s\n', region_names{r}, tem_s{1}, component_name, frequency_names{f_group});
                            concat_pg = concat_pg_activation_ts_allFreqGroups.(frequency_names{f_group});

                            [~, ~, ~, ~, b] = plsregress(concat_bold, concat_pg, pls_n_components);
                            plsbetas = b(2:end, :);

                            %average plsbetas across cells within each frequency
                            num_voxel = size(plsbetas, 1);
                            freqAvg_plsbetas = NaN(num_voxel, length(frequencies{f_group}));
                            cell_start = 1; % starting index for cells
                            for freq_idx = 1:length(frequencies{f_group})
                                num_cells_in_freq = num_cells_per_freq_allFreqGroups.(frequency_names{f_group})(freq_idx);
                                cell_end = cell_start + num_cells_in_freq - 1; % Ending index for this frequency
                                % mean across the cells for this frequency
                                freqAvg_plsbetas(:, freq_idx) = mean(plsbetas(:, cell_start:cell_end), 2, 'omitnan');
                                cell_start = cell_end + 1; % Update start cell index for next frequency
                            end

                            if ~isfield(tv, region_names{r})
                                tv.(region_names{r}) = struct();
                                subjnames.(region_names{r}) = struct();
                            end
                            if ~isfield(tv.(region_names{r}), component_name)
                                tv.(region_names{r}).(component_name) = masked_dat_all_movies.(region_names{r});
                                subjnames.(region_names{r}).(component_name) = [];
                            end
                            if ~isfield(subj_avg, region_names{r})
                                subj_avg.(region_names{r}) = struct();
                            end
                            
                            if tem_s_idx == 1
                                subj_avg.(region_names{r}).(component_name) = freqAvg_plsbetas;
                            else

                                disp(tem_s_idx)
                                disp(tem_s_idx - 1)
                                disp(size(subj_avg))
                                disp(size(freqAvg_plsbetas))
                                subj_avg.(region_names{r}).(component_name) = (subj_avg.(region_names{r}).(component_name) * (tem_s_idx - 1) + freqAvg_plsbetas) / tem_s_idx;
                            end
                            if tem_s_idx == length(walkers) 
                                subjnames.(region_names{r}).(component_name) = [subjnames.(region_names{r}).(component_name), string(subjects{s})];
                            
                                if s == 1
                                    tv.(region_names{r}).(component_name).dat = subj_avg.(region_names{r}).(component_name);
                                    tv.(region_names{r}).(component_name) = replace_empty(tv.(region_names{r}).(component_name));
                                else
                                    % append the new subject's data to the existing data
                                    %tv.(region_names{r}).(component_name).dat = [tv.(region_names{r}).(component_name).dat, subj_avg.(region_names{r}).(component_name)];

                                    temp_tv = masked_dat_all_movies.(region_names{r});
                                    temp_tv.dat = subj_avg.(region_names{r}).(component_name);
                                    temp_tv = replace_empty(temp_tv);
                                    tv.(region_names{r}).(component_name).dat = [tv.(region_names{r}).(component_name).dat temp_tv.dat];
                                end
                            end
                        end
                    end
                end
            end
        end
    catch ME
       fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end

for r = selected_regions
    for component = 1:length(component_names)
        tv.(region_names{r}).(component_names{component}).removed_images = zeros(size(tv.(region_names{r}).(component_names{component}).dat, 2),1);
        tv.(region_names{r}).(component_names{component}) = remove_empty(tv.(region_names{r}).(component_names{component}));
        tv.(region_names{r}).(component_names{component}).dat(tv.(region_names{r}).(component_names{component}).dat==0) = NaN;


        all_plsbetas = tv.(region_names{r}).(component_names{component}).dat;
        temp_obj = tv.(region_names{r}).(component_names{component});

        xyz = temp_obj.volInfo.xyzlist*temp_obj.volInfo.mat(1:3,1:3);
        xyz(temp_obj.removed_voxels,:)=[];
        for i=1:3
            xyz(:,i)=xyz(:,i)+temp_obj.volInfo.mat(i,4);
        end
        
        freqs = frequencies_allcomponents{component}{1};
        for i=1:length(freqs)
            X = all_plsbetas(:,i:length(freqs):end);
            temp_obj.dat = X;
            output_dir_nifti = [output_dir filesep 'nifti'];
            if ~exist(output_dir_nifti, 'dir')
                mkdir(output_dir_nifti);
            end
            st_thr = threshold(ttest(temp_obj),.05,'FDR');
            st_thr.fullpath = [output_dir_nifti filesep 'PLSbetas_FDR05_' region_names{r} '_' component_names{component} '_' freqs{i} '.nii'];
            st_thr.write;
            st_thr = threshold(ttest(temp_obj),.05,'unc');
            st_thr.fullpath = [output_dir_nifti filesep 'PLSbetas_UNC05_' region_names{r} '_' component_names{component} '_' freqs{i} '.nii'];
            st_thr.write;

            X_table = array2table(X);
            X_table.Properties.VariableNames = subjnames.(region_names{r}).(component_names{component});
            xyz_table = array2table(xyz);
            xyz_table.Properties.VariableNames = {'x', 'y', 'z'};
            %concatenate the two tables
            weights_coord_table = [X_table, xyz_table];
            output_dir_csv = [output_dir filesep 'csv'];
            if ~exist(output_dir_csv, 'dir')
                mkdir(output_dir_csv);
            end
            writetable(weights_coord_table, [output_dir_csv filesep 'PLSbetas_xyzCoords_allsubs_' region_names{r} '_' component_names{component} '_' freqs{i} '.csv']);
        end
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

            cat_activation = freq_data.(category_name);

            % Ensure the activation data is a column vector (num_cells x 1)
            if size(cat_activation, 2) ~= 1
                error('Activation data for category "%s" in frequency "%s" must be a column vector.', category_name, freq_key);
            end

            freq_cat_activation = [freq_cat_activation, cat_activation];
        end

        all_activation_data = [all_activation_data; freq_cat_activation];
    end

    % Get the total number of cells across all frequencies
    num_cells = size(all_activation_data, 1);

    pg_activation_ts= zeros(num_timepoints, num_cells);

    for t = 1:num_timepoints
        % Get the weights for each category at time t
        weights = category_rating_ts(t, :);

        weights = weights / sum(weights);

        % Compute the weighted average activation for this time point
        weighted_activation = zeros(num_cells, 1);

        for cat_idx = 1:num_categories
            category_weight = weights(cat_idx);

            category_activation = all_activation_data(:, cat_idx);

            weighted_activation = weighted_activation + category_weight * category_activation;
        end
        pg_activation_ts(t, :) = weighted_activation';
    end
end



