sample_rate = 16000
n_fft = 1024
hop_length = 400
n_mel = 80
min_db = -60
max_db = 40
reduction_factor = 4

batch_size = 32

char_embed_dim = 256
speaker_embed_dim = 256

enc_conv_blocks = 5
enc_conv_dim = 256
enc_conv_dropout = 0.2
enc_lstm_dim = 256
enc_lstm_dropout = 0.1

query_key_dim = 128
value_dim = 512

dec_prenet_dim = 128
dec_prenet_dropout = 0.5
dec_query_conv_dim = 128
dec_query_lstm_dropout = 0.1
dec_query_conv_dropout = 0.1
dec_score_dropout = 0.05
dec_lstm_dim = 512
dec_lstm_dropout = 0.05
dec_conv_dim = 256
dec_conv_dropout = 0.1

init_lr = 0.0005
lr_decay_steps = 10000
lr_decay_rate = 0.9

att_guide_pos_factor = 10
att_guide_decay_steps = 2000
att_guide_rate = 0.1
