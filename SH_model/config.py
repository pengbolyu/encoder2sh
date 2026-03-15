import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join("/home/pengbo.lv/code/img2sh2/eardecoder2sh/data")

encode_path = "/home/pengbo.lv/code/VAE/jaffe_use_inception_rep1_depth_20260312_ear_left_256/img_encoding/encode_matrix256l_20260312_left_img_ear.pt"
file_numbers_path = os.path.join(DATA_DIR, "fileNumbers.mat")
hrtf_path = os.path.join(DATA_DIR, "HUTUBS_matrix_measured9o.mat")
shvec_path = os.path.join(DATA_DIR, "SH_matrix_9o.mat")

log_file = "training_log9o.txt"
run_dir = os.path.join(BASE_DIR, "runs_0314_lrsch_best_val_lsd_res", "SH_model_9o_10f_1v_remove_33_48_mat")

num_epochs = 800
batch_size = 10
learning_rate = 0.001
min_learning_rate = 1e-6
lr_scheduler_factor = 0.5
lr_scheduler_patience = 20
lr_warmup_epochs = 10
weight_decay = 1e-4

# Model architecture
model_stem_channels = 64
model_stage_channels = (128, 256, 512)
model_stage_blocks = (2, 2, 2)

# Ear side selection: 0 for left ear, 1 for right ear
left_or_right = 0

default_seed = 123456
num_individuals = 58
