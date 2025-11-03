set_up_paths_and_data_directories;

types = {'category', 'valence_arousal'};
num_types = length(types);

brain_atlas = load_atlas('canlab2018');
region_masks = {fullfile(folder_project, 'masks', 'HC_Julich.nii.gz'),...
                fullfile(folder_project, 'masks', 'ERC_Julich.nii.gz'),...
                select_atlas_subset(brain_atlas, {'Ctx_10','Ctx_11','Ctx_14','Ctx_25','Ctx_32', 'a24_'}),...
                select_atlas_subset(brain_atlas, {'Amy'}),...
                ''};
region_names = {'Hippocampus', 'EntorhinalCortex', 'vmPFC_a24_included','Amygdala','rest'};
num_regions = length(region_names);

if_save_fisher_z = false;
for t = 1:num_types
    type = types{t};
    display(['Processing type: ', type]);
    %get all files starting with searchlight_mean and ending with .nii
    if if_save_fisher_z
        allsubsfiles = dir(fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', type, 'searchlight', 'searchlight_mean_*.nii'));
    else
        allsubsfiles = dir(fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', type, 'searchlight', 'searchlight_Zmean_*.nii'));
    end
    if isempty(allsubsfiles)
        warning(['No files found for ', type, ' in ', folder_path]);
        continue
    else
        disp(['Found ', num2str(length(allsubsfiles)), ' files.'])
    end

    
    %save fisher z transformed data for each subject
    if if_save_fisher_z
        for s_idx = 1:length(allsubsfiles)
            subdata = fmri_data(fullfile(allsubsfiles(s_idx).folder, allsubsfiles(s_idx).name));
            subdata.dat = atanh(subdata.dat);
            %get subject name from file name
            subject_name = strsplit(allsubsfiles(s_idx).name, '_');
            subject_name = strsplit(subject_name{end}, '.');
            subject_name = subject_name{1};
            subdata.fullpath = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', type, 'searchlight', ['searchlight_Zmean_', subject_name, '.nii']);
            if exist(subdata.fullpath, 'file')
                continue
            end
            subdata.write;
            
        end
    end
    allsubs_data = fmri_data(fullfile({allsubsfiles.folder}, {allsubsfiles.name})');
    %allsubs_data.dat has size of voxels*subs, get cohen's d for the whole brain and each region's average
    d = mean(allsubs_data.dat') ./ std(allsubs_data.dat');
    tmp = allsubs_data;
    tmp.dat = d';
    tmp.fullpath = fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', type, 'searchlight', 'searchlight_d.nii');
    tmp.write;
    d_regions = [];
    in_rest = zeros(size(tmp.dat));
    for r = 1:num_regions
        if ~strcmp(region_names{r}, 'rest')
            masked_data = apply_mask(allsubs_data, region_masks{r});
            in_rest = in_rest + masked_data.removed_voxels;
            masked_data = masked_data.dat;
        else
            masked_data = allsubs_data.dat(in_rest, :);
        end
        region_avg = mean(masked_data);
        d_regions = [d_regions; mean(region_avg)/std(region_avg)];
    end
    d_regions = array2table(d_regions', 'VariableNames', region_names);
    writetable(d_regions, fullfile(folder_project, 'outputs', 'rep3', 'ratings_prediction_performance', 'brain', type, 'searchlight', 'searchlight_d_regions.csv'));
end

