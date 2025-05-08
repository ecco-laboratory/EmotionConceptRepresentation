clear all; close all; clc;
%% Define relevant script-specific constants
folder_project = fileparts(mfilename('fullpath'));

addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks/'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')


% The string after subject, session, task, and other changing items in the BIDS-compatible filename
bold_suffix = 'space-MNI_desc-ppres_bold.nii';
% STUDY-SPECIFIC! SHOULD BE AN ARG
tr_length = 1.3;

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

brain_atlas = load_atlas('canlab2018');

region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz')};%,...
                %select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'})};
region_names = {'Hippocampus', 'EntorhinalCortex'};%, 'vmPFC'};
%{
region_masks = {select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'})};
region_names = {'vmPFC'};
%}
% Brain folder should go directly to the folder containing the subject subfolders
folder_brain_5subs = '/home/data/eccolab/VisionLanguageEncodingEmotion/Code/EmotionConcepts/data/aws/';
subjects_5 = {'sub-S22', 'sub-S25', 'sub-S26', 'sub-S29', 'sub-S32'};
folder_brain_24subs = '/home/data/eccolab/OpenNeuro/ds004892/derivatives/preprocessing/';
subjects_24 = {dir(fullfile(folder_brain_24subs, 'sub-*')).name};
subjects_24 = setdiff(subjects_24, [subjects_5, 'sub-S07']);  %drop S07 and the 5 subjects from the 5 subject set
subjects = [subjects_24, subjects_5];
sessions = {dir(fullfile(folder_brain_24subs, subjects{1}, 'ses-*')).name};

selected_regions = 1:length(region_names);
used_movies = find(~ismember(bids_task_names, {'DamagedKungFu', 'RidingTheRails'}));
num_regions = length(selected_regions);
num_movies = length(used_movies);
num_subjects = length(subjects);

%% CV PLS regression across movies for each region and subject
% Loop through regions and subjects
for s = 1:num_subjects
    if ismember(subjects{s}, subjects_5)
        folder_brain = folder_brain_5subs;
    else
        folder_brain = folder_brain_24subs;
    end
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
        %save a heatmap of the masked data for Hippocampus (jet colormap) to fullfile(folder_project, 'outputs')
        %figure; imagesc(masked_dat_all_movies.Hippocampus.dat'); colormap(jet); 
        %caxis([-150, 150]); colorbar; 
        %set(gca,'XTick',[], 'YTick', []); saveas(gcf, fullfile('/home/data/eccolab/VisionLanguageEncodingEmotion/Code/EmotionConcepts/outputs', 'BOLDheatmap_Hippocampus_sub-S01.png'));

        
        n_trs = [1, n_trs];
        % List of categories to retain

        for type = {'valenceArousal'}%{'category'}%{'binaryValenceArousal'}%%% 'valenceArousal','binaryValenceArousal'}% {'valenceArousal', 'binaryValenceArousal'}%
            if strcmp(type{1}, 'category')
                beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
                emotions = {'Anger', 'Anxiety', 'Fear', 'Surprise', 'Guilt', 'Disgust', ...
                    'Sad', 'Regard', 'Satisfaction', 'WarmHeartedness', 'Happiness', ...
                    'Pride', 'Love'};
                folder_name = 'category';
            elseif strcmp(type{1}, 'binaryValenceArousal')
                beh_data = load(fullfile(folder_project, 'data', 'binary_valence_arousal_beTab.mat'));
                emotions = {'Good', 'Bad', 'Calm', 'AtEase'};
                folder_name = 'binary_valence_arousal';
            elseif strcmp(type{1}, 'valenceArousal')
                beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
                emotions = {'Good', 'Bad', 'Calm', 'AtEase'};
                folder_name = 'valence_arousal';
            end
            behTab = beh_data.behTab;
            % Subset ratings to only include emotion categories
            behTab = structfun(@(tbl) tbl(:, emotions(ismember(emotions, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
            category_names = behTab.(bids_task_names{1}).Properties.VariableNames;
            % Output directory
            output_dir = fullfile(folder_project, 'outputs', 'brain_weight_maps', 'PLSbeta', 'brainToRatings', folder_name);
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
                            all_regions_concat_bold.(region_names{r}) = []; % Initialize the field as an empty array
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
                [~, ~, ~, ~, b] = plsregress(concat_bold, concat_ratings, 20);
                plsbetas = b(2:end, :);
                
                if s==1 & r==1
                    tv = struct();
                    subnames = struct();
                end
                if ~isfield(tv, region_names{r})
                    tv.(region_names{r}) = masked_dat_all_movies.(region_names{r});
                    tv.(region_names{r}).dat = plsbetas;
                    subnames.(region_names{r}) = string(subjects{s});

                    tv.(region_names{r}) = replace_empty(tv.(region_names{r})); 
                    disp(s)
                    disp(region_names{r})
                    disp(size(tv.(region_names{r}).dat))
                else
                    disp(s)
                    disp(region_names{r})
                    disp(size(plsbetas))
                    subnames.(region_names{r}) = [subnames.(region_names{r}), string(subjects{s})];
                    %tv.(region_names{r}).dat = [tv.(region_names{r}).dat plsbetas];

                    temp_tv = masked_dat_all_movies.(region_names{r}); 
                    temp_tv.dat = plsbetas;
                    temp_tv = replace_empty(temp_tv);
                    tv.(region_names{r}).dat = [tv.(region_names{r}).dat temp_tv.dat];
                    
                end
            end
        end
    catch ME
        fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end

%%
for r = selected_regions
    tv.(region_names{r}).removed_images = zeros(size(tv.(region_names{r}).dat, 2),1);
    tv.(region_names{r}) = remove_empty(tv.(region_names{r}));
    tv.(region_names{r}).dat(tv.(region_names{r}).dat==0) = NaN;


    all_plsbetas = tv.(region_names{r}).dat;
    temp_obj = tv.(region_names{r});
    xyz = temp_obj.volInfo.xyzlist*temp_obj.volInfo.mat(1:3,1:3);
    xyz(temp_obj.removed_voxels,:)=[];
    for i=1:3
        xyz(:,i)=xyz(:,i)+temp_obj.volInfo.mat(i,4);
    end

    for i=1:length(emotions)
        X = all_plsbetas(:,i:length(emotions):end);
        temp_obj.dat = X;
        output_dir_nifti = [output_dir filesep 'nifti'];
        if ~exist(output_dir_nifti, 'dir')
            mkdir(output_dir_nifti);
        end
        %save the nifti file
        st_thr = threshold(ttest(temp_obj),.05,'FDR');
        st_thr.fullpath = [output_dir_nifti filesep 'PLSbetas_FDR05_' region_names{r} '_' emotions{i} '.nii'];
        st_thr.write;
        st_thr = threshold(ttest(temp_obj),.05,'unc');
        st_thr.fullpath = [output_dir_nifti filesep 'PLSbetas_UNC05_' region_names{r} '_' emotions{i} '.nii'];
        st_thr.write;

        X_table = array2table(X);
        X_table.Properties.VariableNames = subnames.(region_names{r});
        xyz_table = array2table(xyz);
        xyz_table.Properties.VariableNames = {'x', 'y', 'z'};
        %concatenate the two tables
        weights_coord_table = [xyz_table, X_table];
        %save the table as a csv file
        output_dir_csv = [output_dir filesep 'csv'];
        if ~exist(output_dir_csv, 'dir')
            mkdir(output_dir_csv);
        end
        writetable(weights_coord_table, [output_dir_csv filesep 'PLSbetas_xyzCoords_allsubs_' region_names{r} '_' emotions{i} '.csv']);

    end
end


