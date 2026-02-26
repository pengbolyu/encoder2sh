import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join("/home/pengbo.lv/code/img2sh2/eardecoder2sh/data")

encode_path = os.path.join(DATA_DIR, "encode_matrix256_20241214_left_depth_ear.pt")
file_numbers_path = os.path.join(DATA_DIR, "fileNumbers.mat")
hrtf_path = os.path.join(DATA_DIR, "HUTUBS_matrix_measured8o.mat")
shvec_path = os.path.join(DATA_DIR, "SH_matrix_8o.mat")

log_file = "training_log8o.txt"

num_epochs = 1000
batch_size = 10
learning_rate = 0.001

default_seed = 123456
num_individuals = 58
