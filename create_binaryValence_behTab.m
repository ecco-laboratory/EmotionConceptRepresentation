set_up_paths_and_data_directories;

beh_data = load(fullfile(folder_project, 'data', 'BehavioralRatingsPerVideoAndDim.mat'));
behTab = beh_data.behTab;

emotions = {'Good', 'Bad', 'Calm', 'AtEase'};
behTab = structfun(@(tbl) tbl(:, emotions(ismember(emotions, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
all_ratings = [];  

for movieField = fieldnames(behTab)'
    movieData = behTab.(movieField{1});
    
    for emotion = emotions
        emotionData = movieData{:, emotion};
        all_ratings = [all_ratings; emotionData(~isnan(emotionData))]; % Remove NaNs for min/max calculation
    end
end

ratingMin = min(all_ratings);
ratingMax = max(all_ratings);
middleValue = (ratingMin + ratingMax) / 2;

for movieField = fieldnames(behTab)'
    movieData = behTab.(movieField{1});
    
    for emotion = emotions
        emotionData = movieData{:, emotion};
        
        emotionDataBin = emotionData; % Make a copy to preserve NaNs
        emotionDataBin(emotionData < middleValue & ~isnan(emotionData)) = 0;
        emotionDataBin(emotionData >= middleValue & ~isnan(emotionData)) = 1;
        
        movieData{:, emotion} = emotionDataBin;
    end
    
    behTab.(movieField{1}) = movieData;
end

% The behTab now contains binary ratings in the same format as before.
save(fullfile(folder_project, 'data', 'binary_valence_arousal_behTab.mat'), 'behTab');
