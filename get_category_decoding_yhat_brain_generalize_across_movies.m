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

% Output directory
output_dir = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_yhat');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end


brain_atlas = load_atlas('canlab2018');
region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...%select_atlas_subset(brain_atlas, {'Ctx_H'}),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz'),...%select_atlas_subset(brain_atlas, {'Ctx_EC'}),...
                select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'}),...
                fullfile(folder_project, 'masks', 'HC_ant_Julich.nii'), ...
                fullfile(folder_project, 'masks', 'HC_post_Julich.nii')};
region_names = {'Hippocampus', 'EntorhinalCortex', 'vmPFC_a24_included', 'anteriorHippocampus', 'posteriorHippocampus'};


% Brain folder should go directly to the folder containing the subject subfolders (change this to where you have the data)
subjects = {dir(fullfile(folder_brain, 'sub-*')).name};
subjects = setdiff(subjects, 'sub-S07');  %drop S07 
sessions = {dir(fullfile(folder_brain, subjects{1}, 'ses-*')).name};

selected_regions = 1:length(region_names);
used_movies = find(~ismember(bids_task_names, {'DamagedKungFu', 'RidingTheRails'}));
num_regions = length(selected_regions);
num_movies = length(used_movies);
num_subjects = length(subjects);

emotions_all = {{'Anger', 'Anxiety', 'Fear', 'Surprise', 'Guilt', 'Disgust', ...
   'Sad', 'Regard', 'Satisfaction', 'WarmHeartedness', 'Happiness', ...
   'Pride', 'Love'}, {'Good', 'Bad'}, {'Good', 'Bad', 'Calm','AtEase'},{'Good', 'Bad'}, {'Good', 'Bad', 'Calm', 'AtEase'}};
emotion_types = {'category', 'valence', 'valence_arousal', 'binary_valence', 'binary_valence_arousal'};%{'category', 'binary_valence', 'binary_valence_arousal'};
%{
emotions_all = {{'Good', 'Bad', 'Calm', 'AtEase'}, {'Good', 'Bad', 'Calm', 'AtEase'}};
emotion_types = {'valence_arousal', 'binary_valence_arousal'};%{'category', 'binary_valence', 'binary_valence_arousal'};
%}
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
            mask = region_masks{r}; %select_atlas_subset(load_atlas('canlab2018'), regions{r});
            masked_dat_all_movies.(region_names{r}) = apply_mask(dat_all_movies, mask);
        end

        n_trs = [1, n_trs];
        
        % do category and valence separately
        for e = 1:length(emotions_all)
            fprintf('Running PLS regression for %d---------------------------\n', e)
            emotions = emotions_all{e};

            if strcmp(emotion_types{e}, 'binary_valence_arousal') || strcmp(emotion_types{e}, 'binary_valence')
                beh_data = load(fullfile(folder_project, 'data', 'binary_valence_arousal_beTab.mat'));
            else
                beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
            end
            %% Load behavioral ratings
            behTab = beh_data.behTab;

            % Subset ratings to only include relevant emotions
            behTab = structfun(@(tbl) tbl(:, emotions(ismember(emotions, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
            category_names = behTab.(bids_task_names{1}).Properties.VariableNames;
            

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
            %track time
            tic
            for r = selected_regions
                concat_bold = all_regions_concat_bold.(region_names{r});
                % Run PLS regression 
                clear yhat* pred* diag*
                movie_column = {};
                for k = 1:max(kinds)
                    [~, ~, ~, ~, beta_cv] = plsregress(concat_bold(kinds ~= k, :), concat_ratings(kinds ~= k, :), 20);
                    yhat(kinds == k, :) = [ones(length(find(kinds == k)), 1) concat_bold(kinds == k, :)] * beta_cv;
                    current_available_movie_name = string(bids_task_names{available_movies{k}});
                    movie_column = [movie_column; repmat({current_available_movie_name}, length(find(kinds == k)), 1)];
                end
                yhat_table = array2table(yhat, 'VariableNames', category_names);
                yhat_table.movie = movie_column;
                y_table = array2table(concat_ratings, 'VariableNames', category_names);
                y_table.movie = movie_column;

                % Save the performance data to a CSV file iteratively for each subject and region
                type_output_dir = fullfile(output_dir, emotion_types{e});
                if ~exist(type_output_dir, 'dir')
                    mkdir(type_output_dir);
                end
                yhat_output_file = fullfile(type_output_dir, sprintf('yhat_%s_%s.csv', subjects{s}, region_names{r}));
                y_output_file = fullfile(type_output_dir, sprintf('y_%s_%s.csv', subjects{s}, region_names{r}));

                writetable(yhat_table, yhat_output_file);
                writetable(y_table, y_output_file);
                fprintf('Saved yhat and y tables for subject %s, region %s\n', subjects{s}, region_names{r});
            end
            toc
        end
    catch ME
        fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end



