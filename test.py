
obs_dim = 288
act_dim = 1
mem_pre_lstm_hid_sizes = (128,)
mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)

print(mem_pre_lstm_layer_size)