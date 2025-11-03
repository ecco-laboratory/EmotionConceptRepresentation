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


types = {'category', 'valenceArousal'};%,'binaryValenceArousal'};
if ~exist('types', 'var') || ~iscell(types) || ~ismember(types{1}, {'category', 'valenceArousal', 'binaryValenceArousal'})
    error('types must be one of: {''category''}, {''valenceArousal''}, or {''binaryValenceArousal''}');
end

% Brain folder should go directly to the folder containing the subject subfolders (change this to where you have the data)
subjects = {dir(fullfile(folder_brain, 'sub-*')).name};
subjects = setdiff(subjects, 'sub-S07');  %drop S07 
sessions = {dir(fullfile(folder_brain, subjects{1}, 'ses-*')).name};

used_movies = find(~ismember(bids_task_names, {'DamagedKungFu', 'RidingTheRails'}));
num_movies = length(used_movies);
num_subjects = length(subjects);
searchlight_radius = 4;

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
        searchlight_data = fmri_data(files_all_movies{1});
        
        n_trs = [1, n_trs];

        for type = types% {'valenceArousal', 'binaryValenceArousal'}%
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
            output_dir = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', folder_name, 'searchlight');
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            t_id = 1;
            concat_ratings = [];
            kinds = [];
            concat_bold = [];
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
                    
                    dat_current_movie = dat_all_movies.dat(:, sum(cell2mat(n_trs(1:t))):(sum(cell2mat(n_trs(1:t+1))) - 1));
                    starting_tr = round(90 / tr_length);
                    dat_current_movie = dat_current_movie(:, starting_tr + (1:height(normative_self_report)));
                    dat_current_movie = dat_current_movie';  % Transpose to time x voxels
                    concat_bold = [concat_bold; dat_current_movie];

                catch ME
                    fprintf('Error processing subject %s, movie %s: %s\n', subjects{s}, bids_task_names{t}, ME.message);
                end
            end
            fprintf('Running PLS regression for subject %s\n', subjects{s})
            performance_table_searchlight = run_searchlight(searchlight_data, @decode_emotion, 'r', searchlight_radius, ...
                'concat_bold', concat_bold, 'concat_ratings', concat_ratings, 'kinds', kinds, 'category_names', category_names,...
                'output_dir', output_dir, 'subject', subjects{s});
            %get the average across categories
            performance_table_searchlight = [performance_table_searchlight, mean(performance_table_searchlight, 2, 'omitnan')];
            save(fullfile(output_dir, ...
                    sprintf('performance_table_searchlight_%s.mat', subjects{s})), ...
                    'performance_table_searchlight', '-v7.3');


            %save each column to nifti file
            for var_idx = 1:size(performance_table_searchlight, 2)
                searchlight_data_tmp = searchlight_data;
                searchlight_data_tmp.dat = table2array(performance_table_searchlight(:, var_idx));
                var_name = performance_table_searchlight.Properties.VariableNames{var_idx};
                searchlight_data_tmp.fullpath = fullfile(output_dir, sprintf('searchlight_%s_%s.nii', var_name, subjects{s}));
                searchlight_data_tmp.write;
            end
        end
    catch ME
        fprintf('Error processing subject %s: %s\n', subjects{s}, ME.message);
    end
end


function performance_table = decode_emotion(concat_bold, concat_ratings, kinds, category_names)
    [~, ~, ~, ~, b] = plsregress(concat_bold, concat_ratings, 20);
    clear yhat* pred* diag* pred_obs_corr* diag_corr*;
    for k = 1:max(kinds)
        [~, ~, ~, ~, beta_cv] = plsregress(concat_bold(kinds ~= k, :), concat_ratings(kinds ~= k, :), 20);
        yhat(kinds == k, :) = [ones(length(find(kinds == k)), 1) concat_bold(kinds == k, :)] * beta_cv;
        pred_obs_corr(:, :, k) = corr(yhat(kinds == k, :), concat_ratings(kinds == k, :));
        diag_corr(k, :) = diag(pred_obs_corr(:, :, k));
    end
    performance_table = array2table(mean(diag_corr), 'VariableNames', category_names);
end



function results_table = run_searchlight(searchlight_data, custom_function, varargin)
    % Run searchlight analysis
    %
    % Example usage:
    % results_table = run_searchlight(searchlight_data, @decode_emotion, ...
    %     'r', 5, 'concat_bold', concat_bold, 'concat_ratings', concat_ratings, ...
    %     'kinds', kinds, 'category_names', category_names);
    
        % -------------------------
        % Parse inputs
        % -------------------------
        r = 3;
        indx = [];
        concat_bold = [];concat_ratings = [];kinds = [];category_names = [];output_dir = pwd; subject = 'unknown';
        xyz = searchlight_data.volInfo.xyzlist;  % voxel coordinates (nvox × 3)
    
        for i = 1:length(varargin)
            if ischar(varargin{i})
                switch varargin{i}
                    case 'r'
                        r = varargin{i+1};
                    case 'indx'
                        indx = varargin{i+1};
                    case 'concat_bold'
                        concat_bold = varargin{i+1};
                    case 'concat_ratings'
                        concat_ratings = varargin{i+1};
                    case 'kinds'
                        kinds = varargin{i+1};
                    case 'category_names'
                        category_names = varargin{i+1};
                    case 'output_dir'
                        output_dir = varargin{i+1};
                    case 'subject'
                        subject = varargin{i+1};
                end
            end
        end

        nvox = searchlight_data.volInfo.n_inmask;
        if size(concat_bold, 2) ~= nvox
            error('Voxel number mismatch between concat_bold (%d) and searchlight_data (%d).', size(concat_bold,2), nvox);
        end
    
        % -------------------------
        % Build sphere indices
        % -------------------------
        %if isempty(indx)
        %    indx = searchlight_sphere_prep(searchlight_data, r);
        %else
        %    fprintf('Using provided searchlight spheres...\n');
        %end
    
        % -------------------------
        % Estimate runtime
        % -------------------------
        fprintf('Estimating runtime...\n');
        n_to_run = min(100, nvox);
        tic;
        tmp_cell = cell(n_to_run, 1);
        for v = 1:n_to_run
            center = xyz(v, :);
            sphere_vox = sum((xyz - center).^2, 2) <= r^2;
            if ~any(sphere_vox), continue; end
            sphere_data = concat_bold(:, sphere_vox); % time × voxels
            try
                tmp_cell{v} = custom_function(sphere_data, concat_ratings, kinds, category_names);
            catch
                tmp_cell{v} = [];
            end
        end
        elapsed = toc;
        estim_total = elapsed * (nvox / n_to_run);
        [hour, minute, second] = sec2hms(estim_total);
        fprintf('Estimated total runtime: %d hr %d min %.1f sec\n', hour, minute, second);
    
        % -------------------------
        % Run full searchlight
        % -------------------------
        fprintf('Running searchlight with %d voxels...\n', nvox);
        outcell = cell(nvox, 1);
        tic;
        update_every = 3000;
        for v = 1:nvox
            center = xyz(v, :);
            sphere_vox = sum((xyz - center).^2, 2) <= r^2;
            if ~any(sphere_vox), continue; end
            sphere_data = concat_bold(:, sphere_vox); % time × voxels
            try
                outcell{v} = custom_function(sphere_data, concat_ratings, kinds, category_names);
            catch
                outcell{v} = [];
            end

            if mod(v, update_every)==0 || v==nvox
                elapsed = toc;
                pct = 100 * v / nvox;
                avg_time = elapsed / v;
                remaining = avg_time * (nvox - v);
                fprintf('%.1f%% complete | elapsed %.1f min | remaining %.1f min\n', pct, elapsed/60, remaining/60);
            end
        end
        elapsed = toc;
        [hour, minute, second] = sec2hms(elapsed);
        fprintf('Done in %d hr %d min %.1f sec\n', hour, minute, second);
    
        % -------------------------
        % Combine to a table
        % -------------------------
        first_valid = find(~cellfun(@isempty,outcell), 1);
        if isempty(first_valid), error('All spheres failed.'); end
        varnames = outcell{first_valid}.Properties.VariableNames;
        nvars = numel(varnames);
    
        results_mat = nan(nvox, nvars);
        nan_row = nan(1, nvars);
    
        for v = 1:nvox
            if isempty(outcell{v})
                results_mat(v, :) = nan_row;
            else
                row = table2array(outcell{v});
                if numel(row) ~= nvars
                    results_mat(v, :) = nan_row;
                else
                    results_mat(v, :) = row;
                end
            end
        end
    
        results_table = array2table(results_mat, 'VariableNames', varnames);
    end
    
    
% -------------------------
% Helper: searchlight spheres
% -------------------------
function indx = searchlight_sphere_prep(dat, r)
    
    nvox = dat.volInfo.n_inmask;
    indx = cell(1, nvox);
    
    t=tic;
    fprintf('Preparing %d seeds...\n', nvox);
    
    for i = 1:nvox
        seed{i} = dat.volInfo.xyzlist(i, :);
    end
    e = toc(t);
    fprintf('Done in %3.2f sec\n', e);

    % Set up indices for spherical searchlight
    % -------------------------------------------------------------------------
    % These could be indices for ROIs, user input, previously saved indices...

    % First, a rough time estimate:
    % -------------------------------------------------------------------------
    fprintf('Searchlight sphere construction can take 20 mins or more! (est: 20 mins with 8 processors/gray matter mask)\n');
    fprintf('It can be re-used once created for multiple analyses with the same region definitions\n');
    fprintf('Getting a rough time estimate for how long this will take...\n');

    n_to_run = min(500, nvox);
    t = tic;
    for i = 1:n_to_run
        
        mydist = sum([dat.volInfo.xyzlist(:, 1) - seed{i}(1) dat.volInfo.xyzlist(:, 2) - seed{i}(2) dat.volInfo.xyzlist(:, 3) - seed{i}(3)] .^ 2, 2);
        indx{i} = mydist <= r.^2;
        
    end
    e = toc(t);
    estim = e * nvox / n_to_run;

    [hour, minute, second] = sec2hms(estim);
    fprintf(1,'\nEstimate for whole brain = %3.0f hours %3.0f min %2.0f sec\n',hour, minute, second);

    % Second, do it for all voxels/spheres:
    % -------------------------------------------------------------------------
    t = tic;
    fprintf('Constructing spheres...\n');
    for i = 1:nvox
        mydist = sum([dat.volInfo.xyzlist(:,1)-seed{i}(1), ...
                      dat.volInfo.xyzlist(:,2)-seed{i}(2), ...
                      dat.volInfo.xyzlist(:,3)-seed{i}(3)].^2, 2);
        indx{i} = mydist <= r^2;
    end
    e = toc(t);
    fprintf('Done in %3.2f sec\n', e);

    %Make sparse matrix
    t = tic;
    fprintf('Making sparse matrix...\n');
    indx = sparse(cat(2, indx{:}));

    e = toc(t);
    fprintf('Done in %3.2f sec\n', e);
    
end %end of function searchlight_sphere_prep
    
    
% -------------------------
% Helper: seconds → h/m/s
% -------------------------
function [hour, minute, second] = sec2hms(sec)
    hour   = fix(sec/3600);
    sec    = sec - 3600*hour;
    minute = fix(sec/60);
    sec    = sec - 60*minute;
    second = sec;
end