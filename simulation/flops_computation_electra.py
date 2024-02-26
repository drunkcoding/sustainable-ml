#Reference: https://github.com/google-research/electra/blob/master/flops_computation.py
import collections
import abc
# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class FLOPS_Calculation(metaclass=abc.ABCMeta):
	def __init__(self, h, s, v=30522, e=None, i=None, heads=None, head_size=None, 
			output_frac=0.15625, sparse_embed_lookup=False,
			decoder=False):
		self.h = h
		self.s = s
		self.v = v
		self.e = h if e is None else e
		self.i = 4*h if i is None else i
		self.kqv = h if head_size is None else head_size * heads
		self.heads = max(h//64, 1) if heads is None else heads
		self.output_frac = output_frac
		self.sparse_embed_lookip = sparse_embed_lookup
		self.decoder = decoder

	def get_block_flops(self):
		attn_mul = 2 if self.decoder else 1
		block_flops = dict(
					kqv=3 * 2 * self.h * self.kqv * attn_mul,
					kqv_bias=3 * self.kqv * attn_mul,
					attention_scores=2 * self.kqv * self.s * attn_mul,
					attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
					attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
					attention_scale=self.s * self.heads * attn_mul,
					attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
					attn_output=2 * self.h * self.h * attn_mul,
					attn_output_bias=self.h * attn_mul,
					attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
					attn_output_residual=self.h * attn_mul,
					attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
					intermediate=2 * self.h * self.i,
					intermediate_act=ACTIVATION_FLOPS * self.i,
					intermediate_bias=self.i,
					output=2 * self.h * self.i,
					output_bias=self.h,
					output_dropout=DROPOUT_FLOPS * self.h,
					output_residual=self.h,
					output_layer_norm=LAYER_NORM_FLOPS * self.h,
				)
		return sum(block_flops.values()) * self.s


	def get_embedding_flops(self, output=False):
		"""Get the forward-pass FLOPs the transformer inputs or output softmax."""
		embedding_flops = {}
		if output or (not self.sparse_embed_lookup):
			embedding_flops["main_multiply"] = 2 * self.e * self.v
		if not output:
			embedding_flops.update(dict(
				tok_type_and_position=2 * self.e * (self.s + 2),
				add_tok_type_and_position=2 * self.e,
				emb_layer_norm=LAYER_NORM_FLOPS * self.e,
				emb_dropout=DROPOUT_FLOPS * self.e
			))
		# projection layer if e != h
		if self.e != self.h or output:	
			embedding_flops.update(dict(
			hidden_kernel=2 * self.h * self.e,
			hidden_bias=self.e if output else self.h
			))	
			if output:
				embedding_flops.update(dict(
				hidden_activation=ACTIVATION_FLOPS * self.e,
				hidden_layernorm=LAYER_NORM_FLOPS * self.e,
				output_softmax=SOFTMAX_FLOPS * self.v,
				output_target_word=2 * self.v
			))
			return self.output_frac * sum(embedding_flops.values()) * self.s
		return sum(embedding_flops.values()) * self.s


	def get_train_flops_per_layer(self, batch_size):
		return 2*batch_size*self.get_block_flops()

	def get_train_flops_final_layer(self, batch_size):
		return batch_size*(self.get_embedding_flops(output=False) +
			self.get_embedding_flops(output=True))
