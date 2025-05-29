clear all; close all; clc;

folder_project = fileparts(mfilename('fullpath'));
%% Add paths (change this to where you have the dependencies installed)
addpath(genpath('/home/data/eccolab/Code/GitHub/Neuroimaging_Pattern_Masks/'))
addpath(genpath('/home/data/eccolab/Code/GitHub/CanlabCore'))
addpath('/home/data/eccolab/Code/GitHub/spm12')



components = {'p', 'g'};
freqs = {'freq4', 'freq3', 'freq2', 'freq1', 'freq0'};
regions = {'Hippocampus', 'EntorhinalCortex', 'vmPFC'};
map_dir = fullfile(folder_project, 'outputs', 'brain_weight_maps', 'PLSbeta', 'brainToTEM', 'MDSseed121', 'iteration32000', 'walkRandomSeed42', 'nifti');
thres = {'FDR05', 'UNC05'};

for i = 1:length(regions)
    region = regions{i};
    for j = 1:length(components)
        component = components{j};
        for k = 1:length(thres)
            threshold = thres{k};

            file_dirs = dir(fullfile(map_dir, ['PLSbetas_', threshold, '_', region, '_', component, '*.nii']));
            file_names = {file_dirs.name};

            % sort files according to the frequency order
            sorted_files = cell(1, length(freqs));
            for f = 1:length(freqs)
                match_idx = find(contains(file_names, freqs{f}), 1);
                if isempty(match_idx)
                    error('Missing frequency file: %s for %s, %s, %s', freqs{f}, region, component, threshold);
                end
                sorted_files{f} = fullfile(file_dirs(match_idx).folder, file_dirs(match_idx).name);
            end

            all_data = [];
            for f = 1:length(sorted_files)
                fmri_obj = fmri_data(sorted_files{f});
                all_data(:, f) = fmri_obj.dat(:); % voxel x frequency matrix
            end

            zero_mask = all(all_data == 0, 2);

            % index of the max value for each voxel
            [~, max_freq_idx] = max(all_data, [], 2);

            % assign 0 to voxels that were zero across all frequencies
            max_freq_idx(zero_mask) = 0;

            output_filename = fullfile(map_dir, ['MaxFreqIdx_', threshold, '_', region, '_', component, '.nii']);
            fmri_obj.dat = max_freq_idx; % replace data with frequency index
            write(fmri_obj, 'fname', output_filename);

            fprintf('Saved max frequency index map: %s\n', output_filename);
        end
    end
end
