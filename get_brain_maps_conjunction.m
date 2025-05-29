clear all; close all; clc;

%% Add paths (change this to where you have the dependencies installed)
folder_project = fileparts(mfilename('fullpath'));
addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks/'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')


emo_types = {'category','binary_valence_arousal', 'valence_arousal'};
emotions_category = {'Anger', 'Anxiety', 'Fear', 'Surprise', 'Guilt', 'Disgust', ...
                    'Sad', 'Regard', 'Satisfaction', 'WarmHeartedness', 'Happiness', ...
                    'Pride', 'Love'};
emotions_valence_arousal = {'Good', 'Bad', 'Calm', 'AtEase'};
regions = {'Hippocampus', 'EntorhinalCortex','vmPFC'};
map_dir = fullfile(folder_project,'outputs','brain_weight_maps','PLSbeta','brainToRatings');
thres = {'FDR05','UNC05'};
for i = 1:length(regions)
    region = regions{i};
    for j = 1:length(emo_types)
        emo_type = emo_types{j};
        if strcmp(emo_type, 'category')
            emotions = emotions_category;
        else
            emotions = emotions_valence_arousal;
        end
        for k = 1:length(thres)
            threshold = thres{k};
            % e.g., PLSbetas_UNC05_Hippocampus_Guilt.nii
            file_dirs = dir(fullfile(map_dir, emo_type,'nifti',['PLSbetas_',threshold,'_',region,'*.nii']));
            file_dirs = file_dirs(contains({file_dirs.name}, emotions));
            %create conjunction map (normalized number of of emotions that have value in each voxel, separately for negative and positive values)
            for l = 1:length(file_dirs)
                if l == 1
                    map = fmri_data(fullfile(map_dir, emo_type,'nifti',file_dirs(l).name));
                    map_dat_pos = double(map.dat > 0);
                    map_dat_neg = double(map.dat < 0);
                else
                    map = fmri_data(fullfile(map_dir, emo_type,'nifti',file_dirs(l).name));
                    map_dat_pos = map_dat_pos + double(map.dat > 0);
                    map_dat_neg = map_dat_neg + double(map.dat < 0);
                end
            end
            map_dat_pos = map_dat_pos/length(file_dirs);
            map_dat_neg = -map_dat_neg/length(file_dirs);
            map.dat = map_dat_pos + map_dat_neg;
            map.fullpath = [map_dir filesep emo_type filesep 'nifti' filesep 'PLSbetas_' threshold '_' region '_conjunction.nii'];
            disp(['Writing ' map.fullpath])
            map.write;
        end
    end
end
 
          
