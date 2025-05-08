% Filename: transform_ratings_to_binary.m

% Load data
beh_data = load('/home/data/eccolab/VisionLanguageEncodingEmotion/BehavioralRatingsPerVideoAndDim.mat');
behTab = beh_data.behTab;

% Define emotions and initialize
emotions = {'Good', 'Bad', 'Calm', 'Alert', 'AtEase'};
behTab = structfun(@(tbl) tbl(:, emotions(ismember(emotions, tbl.Properties.VariableNames))), behTab, 'UniformOutput', false);
all_ratings = [];  % To store all ratings for determining middle value

% Collect all ratings for Good and Bad across all movies
for movieField = fieldnames(behTab)'
    movieData = behTab.(movieField{1});
    
    % Extract ratings for Good and Bad columns (considering NaNs)
    for emotion = emotions
        emotionData = movieData{:, emotion};
        all_ratings = [all_ratings; emotionData(~isnan(emotionData))]; % Remove NaNs for min/max calculation
    end
end

% Find the minimum and maximum ratings for Good and Bad across all movies
ratingMin = min(all_ratings);
ratingMax = max(all_ratings);

% Calculate the middle value
middleValue = (ratingMin + ratingMax) / 2;

% Now apply this threshold to create binary ratings
for movieField = fieldnames(behTab)'
    movieData = behTab.(movieField{1});
    
    % Apply the transformation to each emotion column
    for emotion = emotions
        emotionData = movieData{:, emotion};
        
        % Transform ratings
        emotionDataBin = emotionData; % Make a copy to preserve NaNs
        emotionDataBin(emotionData < middleValue & ~isnan(emotionData)) = 0;
        emotionDataBin(emotionData >= middleValue & ~isnan(emotionData)) = 1;
        
        % Assign back to the table
        movieData{:, emotion} = emotionDataBin;
    end
    
    % Update the movie field in behTab
    behTab.(movieField{1}) = movieData;
end

% The behTab now contains binary ratings in the same format as before.
save('/home/data/eccolab/VisionLanguageEncodingEmotion/Code/EmotionConcepts/outputs/binary_valence_arousal_beTab.mat', 'behTab');
