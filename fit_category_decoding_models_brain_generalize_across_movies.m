set_up_paths_and_data_directories;

bold_suffix = 'space-MNI_desc-ppres_bold.nii';
tr_length = 1.3;

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

brain_atlas = load_atlas('canlab2018');
region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz'),...
                select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'}),...
                fullfile(folder_project, 'masks', 'HC_ant_Julich.nii'), ...
                fullfile(folder_project, 'masks', 'HC_post_Julich.nii')};
region_names = {'Hippocampus', 'EntorhinalCortex', 'vmPFC_a24_included', 'anteriorHippocampus', 'posteriorHippocampus'};



% Brain folder should go directly to the folder containing the subject subfolders 
subjects = {dir(fullfile(folder_brain, 'sub-*')).name};
subjects = setdiff(subjects, 'sub-S07');  %drop S07 
sessions = {dir(fullfile(folder_brain, subjects{1}, 'ses-*')).name};

selected_regions = 1:length(region_names);
used_movies = find(~ismember(bids_task_names, {'DamagedKungFu', 'RidingTheRails'}));
num_regions = length(selected_regions);
num_movies = length(used_movies);
num_subjects = length(subjects);

%% CV PLS regression across movies for each region and subject
% Loop through regions and subjects
for s = 1:num_subjects
      try 
        files_all_movies = {};
        n_trs = {};
        available_movies = {};
        for t = used_movies
            task = bids_task_names{t};
            for session = sessions
                % Construct file path for the current subject and movie task
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
                    n_trs = [n_trs(1:end-1), {0}, n_trs(end)];
                end
            end
        end
        %Load the fMRI data
        fprintf('Loading fmri data from all movies for subject %s\n', subjects{s})
        dat_all_movies = fmri_data(files_all_movies);
        masked_dat_all_movies = struct();
        for r = selected_regions
            % Apply the mask
            mask = region_masks{r} %select_atlas_subset(load_atlas('canlab2018'), regions{r});
            masked_dat_all_movies.(region_names{r}) = apply_mask(dat_all_movies, mask);
        end
        clear dat_all_movies;
        
        n_trs = [1, n_trs];

        for type = {'category', 'valenceArousal','binaryValenceArousal'}% {'valenceArousal', 'binaryValenceArousal'}%
            if strcmp(type{1}, 'category')
                beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
                emotions = {'Anger', 'Anxiety', 'Fear', 'Surprise', 'Guilt', 'Disgust', ...
                    'Sad', 'Regard', 'Satisfaction', 'WarmHeartedness', 'Happiness', ...
                    'Pride', 'Love'};
                folder_name = 'category';
            elseif strcmp(type{1}, 'valenceArousal')
                beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
                emotions = {'Good', 'Bad', 'Calm', 'AtEase'};
                folder_name = 'valence_arousal';
            elseif strcmp(type{1}, 'binaryValenceArousal')
                beh_data = load(fullfile(folder_project, 'data', 'binary_valence_arousal_beTab.mat'));
                emotions = {'Good', 'Bad', 'Calm', 'AtEase'};
                folder_name = 'binary_valence_arousal';
            end
            behTab = beh_data.behTab;
            % Subset ratings to only include relevant emotions
            behTab = structfun(@(tbl) tbl(:, emotions(ismember(emotions, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
            category_names = behTab.(bids_task_names{1}).Properties.VariableNames;
            % Output directory
            output_dir = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', folder_name);
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            t_id = 1;
            concat_ratings = [];
            kinds = [];
            all_regions_concat_bold = struct();
            fprintf('Concatenating data for subject %s\n', subjects{s})
            for t = available_movies
                t=t{1};
                subject = subjects{s};
                task = bids_task_names{t};
                try
                    % Load the behavioral ratings and resample to match BOLD TR
                    normative_self_report = table2array(behTab.(bids_task_names{t}));
                    normative_self_report = fillmissing(normative_self_report, 'nearest', 1);
                    normative_self_report = resample(double(normative_self_report), 10, 13);  % Resample for BOLD TR
                    
                    % Convolve features to match hemodynamic BOLD data
                    for i = 1:size(normative_self_report, 2)
                        tmp = conv(double(normative_self_report(:, i)), spm_hrf(tr_length));
                        conv_ratings(:, i) = tmp(:); 
                        clear tmp;
                    end
                    conv_ratings = conv_ratings(1:height(normative_self_report), :);
                    concat_ratings = [concat_ratings; conv_ratings];
                    clear conv_ratings;

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
            fprintf('Running PLS regression for subject %s\n', subjects{s})
            for r = selected_regions
                concat_bold = all_regions_concat_bold.(region_names{r});
                % Run PLS regression and compute performance
                [~, ~, ~, ~, b] = plsregress(concat_bold, concat_ratings, 20);
                clear yhat* pred* diag* pred_obs_corr* diag_corr*;
                for k = 1:max(kinds)
                    [~, ~, ~, ~, beta_cv] = plsregress(concat_bold(kinds ~= k, :), concat_ratings(kinds ~= k, :), 20);
                    yhat(kinds == k, :) = [ones(length(find(kinds == k)), 1) concat_bold(kinds == k, :)] * beta_cv;
                    pred_obs_corr(:, :, k) = corr(yhat(kinds == k, :), concat_ratings(kinds == k, :));
                    diag_corr(k, :) = diag(pred_obs_corr(:, :, k));
                end
                performance_table = array2table(mean(diag_corr), 'VariableNames', category_names)
                performance_table.subject = string(subjects{s});
                performance_table.region = string(region_names{r});

                % Save the performance data to a CSV file iteratively for each subject and region
                if strcmp(region_names{r}, 'anteriorHippocampus') || strcmp(region_names{r}, 'posteriorHippocampus')
                    output_file = fullfile(output_dir, sprintf('%sRatings_prediction_performance_generalized_across_movies_apHC.csv', type{1}));
                else
                    output_file = fullfile(output_dir, sprintf('%sRatings_prediction_performance_generalized_across_movies_hcecvmpfc.csv', type{1}));
                end
                % Check if the file already exists
                if isfile(output_file)
                    existing_data = readtable(output_file);
                    updated_data = [existing_data; performance_table];
                else
                    % If the file does not exist, the updated data is just the new table
                    updated_data = performance_table;
                end
                writetable(updated_data, output_file);
                fprintf('Performance data for subject %s and region %s saved to %s\n', subjects{s}, region_names{r}, output_file);
            end
        end
    catch ME
        fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end


