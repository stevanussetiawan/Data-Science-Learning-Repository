# Melatih model MDN-RNN

# Impor library

import numpy as np
import os
import json
import time
from vae import reset_graph
from rnn import HyperParams, MDNRNN

# Mengatur indeks GPU apa (jika tersedia) untuk digunakan dalam proses pelatihan
os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

# Menentukan variabel DATA_DIR yang menunjuk ke folder tempat data pelatihan RNN disimpan
DATA_DIR = "series"
# Memeriksa apakah ada folder penyimpanan untuk bobot RNN, jika tidak, folder tersebut akan dibuat
model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

# Memeriksa apakah folder yang menyimpan vektor laten ada, dan jika tidak, membuatnya
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

# Membuat fungsi yang mengembalikan sekumpulan latent vector dan action secara acak
def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

# Membuat fungsi yang mengembalikan semua hyperparameter default model MDN-RNN

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=999,
                     input_seq_width=35,
                     output_seq_width=32,
                     rnn_size=256,
                     batch_size=100,
                     grad_clip=1.0,
                     num_mixture=5,
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

# Mendapatkan dan mengambil sampel hyperparameter default ini

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)
raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# Membagi data pelatihan menjadi tiga bagian spesifik yang terpisah (mu, logvar, dan action)

data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]

# Mengatur hyperparameter yang digunakan untuk pengelompokan data dan pelatihan model

max_seq_len = hps_model.max_seq_len
N_data = len(data_mu)
batch_size = hps_model.batch_size

# Menyimpan 1000 mu awal dan logvar dari pemisahan data di atas

initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open('initial_z.json', 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

# Menyetel ulang grafik model MDN-RNN

reset_graph()

# Membuat model MDN-RNN sebagai objek kelas MDNRNN dengan semua hyperparameter default

rnn = MDNRNN(hps_model)

# Mengimplementasikan Training Loop

hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):
  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
  raw_z, raw_a = random_batch()
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :]
  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "Step: %d, Learning Rate: %.6f, Cost: %.4f, Training Time: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# Menyimpan bobot model MDN-RNN ke dalam file json

rnn.save_json(os.path.join(model_save_path, "rnn.json"))
