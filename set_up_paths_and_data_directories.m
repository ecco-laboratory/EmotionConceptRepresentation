%% clear workspace 
clear all; close all; clc;

%% find the location of this code
folder_project = fileparts(mfilename('set_up_paths_and_data_directories.m'));

%% Add paths (change this to where you have the dependencies installed)
addpath(genpath('C:\Users\pkragel\OneDrive - Emory\Documents\GitHub\Neuroimaging_Pattern_Masks'))
addpath(genpath('C:\Users\pkragel\OneDrive - Emory\Documents\GitHub\CanlabCore'))
addpath 'C:\Users\pkragel\OneDrive - Emory\Documents\spm12\'


%% specify locations of specific data directories
folder_brain = '/home/data/eccolab/OpenNeuro/ds004892/derivatives/preprocessing/'; %location of data downloaded from OpenNeuro

