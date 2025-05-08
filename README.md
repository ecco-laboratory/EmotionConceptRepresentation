1. Download EmoFilM fMRI data
aws s3 sync --no-sign-request s3://openneuro.org/ds004892/derivatives/preprocessing/ path_to_project_folder/data/fmri

2. Train TEM model
python run.py --env_file ./envs/11x11_square_obs45_act5.json --n_train_iterations 1000000 --load_existing_model False

