# ============================================================
# Transformer Configuration
# ============================================================

# These vocabulary values are only defaults.
# The actual vocabulary size is obtained from the tokenizer.
src_vocab_size = 50260
tgt_vocab_size = 50260

# Transformer dimensions
d_model = 128

# Number of attention heads
num_heads = 4

# Number of encoder/decoder layers
num_layers = 2

# Feed-forward network dimension
d_ff = 256

# Maximum sequence length supported by the model
max_length_seq = 32

# Dropout value kept for future use
dropout = 0.1