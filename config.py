src_vocab_size = 5000 # (Total unique words in source language — English)
tgt_vocab_size = 5000 # (Total unique words in target language — German)

d_model = 512 # (Embedding dimension — size of each word vector)
num_heads = 8 # (Number of parallel attention mechanisms)
num_layers = 6 # (Number of encoder and decoder blocks to stack)
d_ff = 2048 # (Feed-forward network hidden layer size)
max_length_seq = 100 # (Maximum number of words per sentence)
dropout = 0.1 # (Percentage of neurons to randomly drop — prevents overfitting)