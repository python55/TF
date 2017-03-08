#--------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
#from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

tf.reset_default_graph()

hm_epochs = 1 # could make this bigger for better accuracy if you wanna fire up your GTX 1070
n_classes = 10 # 0-9 outputs
batch_size = 128 
chunk_size = 28 # 28 pixels in each slice
n_chunks = 28 # have to recurse 28 times (temporal)
rnn_size = 8 # so far 1 deep layer with 8 LSTM cells, could increase if you wanna.
logs_path = '/tmp/tensorflow_logs/example'

x = tf.placeholder('float', [None, n_chunks,chunk_size]) # A placeholder is simply 'reserving seats' in the graph/model.
y = tf.placeholder('float')
#i = tf.placeholder('float')


###### I imported the relavent parts from rnn.py and rnn_cell.py for manipulation

#--------------------------------------------------------------------------------
# rnn.py imports
#--------------------------------------------------------------------------------

def _state_size_with_prefix(state_size, prefix=None):
	"""Helper function that enables int or TensorShape shape specification.

	This function takes a size specification, which can be an integer or a
	TensorShape, and converts it into a list of integers. One may specify any
	additional dimensions that precede the final state size specification.

	Args:
		state_size: TensorShape or int that specifies the size of a tensor.
		prefix: optional additional list of dimensions to prepend.

	Returns:
		result_state_size: list of dimensions the resulting tensor size.
	"""
	result_state_size = tensor_shape.as_shape(state_size).as_list()
	if prefix is not None:
		if not isinstance(prefix, list):
			raise TypeError("prefix of _state_size_with_prefix should be a list.")
		result_state_size = prefix + result_state_size
	return result_state_size
#---------------------------------------------------------------------


# pylint: disable=protected-access
###_state_size_with_prefix = rnn_cell._state_size_with_prefix
_state_size_with_prefix = _state_size_with_prefix
# pylint: enable=protected-access


def rnn(cell, inputs, initial_state=None, dtype=None,
				sequence_length=None, scope=None):
	"""Creates a recurrent neural network specified by RNNCell `cell`.

	The simplest form of RNN network generated is:

	```python
		state = cell.zero_state(...)
		outputs = []
		for input_ in inputs:
			output, state = cell(input_, state)
			outputs.append(output)
		return (outputs, state)
	```
	However, a few other options are available:

	An initial state can be provided.
	If the sequence_length vector is provided, dynamic calculation is performed.
	This method of calculation does not compute the RNN steps past the maximum
	sequence length of the minibatch (thus saving computational time),
	and properly propagates the state at an example's sequence length
	to the final state output.

	The dynamic calculation performed is, at time `t` for batch row `b`,

	```python
		(output, state)(b, t) =
			(t >= sequence_length(b))
				? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
				: cell(input(b, t), state(b, t - 1))
	```

	Args:
		cell: An instance of RNNCell.
		inputs: A length T list of inputs, each a `Tensor` of shape
			`[batch_size, input_size]`, or a nested tuple of such elements.
		initial_state: (optional) An initial state for the RNN.
			If `cell.state_size` is an integer, this must be
			a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
			If `cell.state_size` is a tuple, this should be a tuple of
			tensors having shapes `[batch_size, s] for s in cell.state_size`.
		dtype: (optional) The data type for the initial state and expected output.
			Required if initial_state is not provided or RNN state has a heterogeneous
			dtype.
		sequence_length: Specifies the length of each sequence in inputs.
			An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
		scope: VariableScope for the created subgraph; defaults to "RNN".

	Returns:
		A pair (outputs, state) where:

		- outputs is a length T list of outputs (one for each input), or a nested
			tuple of such elements.
		- state is the final state

	Raises:
		TypeError: If `cell` is not an instance of RNNCell.
		ValueError: If `inputs` is `None` or an empty list, or if the input depth
			(column size) cannot be inferred from inputs via shape inference.
	"""
	###if not isinstance(cell, rnn_cell.RNNCell):
	if not isinstance(cell, RNNCell):
		raise TypeError("cell must be an instance of RNNCell")
	if not nest.is_sequence(inputs):
		raise TypeError("inputs must be a sequence")
	if not inputs:
		raise ValueError("inputs must not be empty")

	outputs = []
	# Create a new scope in which the caching device is either
	# determined by the parent scope, or is set to place the cached
	# Variable using the same placement as for the rest of the RNN.
	with vs.variable_scope(scope or "RNN") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		# Obtain the first sequence of the input
		first_input = inputs
		while nest.is_sequence(first_input):
			first_input = first_input[0]

		# Temporarily avoid EmbeddingWrapper and seq2seq badness
		# TODO(lukaszkaiser): remove EmbeddingWrapper
		if first_input.get_shape().ndims != 1:

			input_shape = first_input.get_shape().with_rank_at_least(2)
			fixed_batch_size = input_shape[0]

			flat_inputs = nest.flatten(inputs)
			for flat_input in flat_inputs:
				input_shape = flat_input.get_shape().with_rank_at_least(2)
				batch_size, input_size = input_shape[0], input_shape[1:]
				fixed_batch_size.merge_with(batch_size)
				for i, size in enumerate(input_size):
					if size.value is None:
						raise ValueError(
								"Input size (dimension %d of inputs) must be accessible via "
								"shape inference, but saw value None." % i)
		else:
			fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

		if fixed_batch_size.value:
			batch_size = fixed_batch_size.value
		else:
			batch_size = array_ops.shape(first_input)[0]
		if initial_state is not None:
			state = initial_state
		else:
			if not dtype:
				raise ValueError("If no initial_state is provided, "
												 "dtype must be specified")
			state = cell.zero_state(batch_size, dtype)

		if sequence_length is not None:  # Prepare variables
			sequence_length = ops.convert_to_tensor(
					sequence_length, name="sequence_length")
			if sequence_length.get_shape().ndims not in (None, 1):
				raise ValueError(
						"sequence_length must be a vector of length batch_size")
			def _create_zero_output(output_size):
				# convert int to TensorShape if necessary
				size = _state_size_with_prefix(output_size, prefix=[batch_size])
				output = array_ops.zeros(
						array_ops.pack(size), _infer_state_dtype(dtype, state))
				shape = _state_size_with_prefix(
						output_size, prefix=[fixed_batch_size.value])
				output.set_shape(tensor_shape.TensorShape(shape))
				return output

			output_size = cell.output_size
			flat_output_size = nest.flatten(output_size)
			flat_zero_output = tuple(
					_create_zero_output(size) for size in flat_output_size)
			zero_output = nest.pack_sequence_as(structure=output_size,
																					flat_sequence=flat_zero_output)

			sequence_length = math_ops.to_int32(sequence_length)
			min_sequence_length = math_ops.reduce_min(sequence_length)
			max_sequence_length = math_ops.reduce_max(sequence_length)

		for time, input_ in enumerate(inputs):
			if time > 0: varscope.reuse_variables()
			# pylint: disable=cell-var-from-loop
			call_cell = lambda: cell(input_, state)
			# pylint: enable=cell-var-from-loop
			if sequence_length is not None:
				(output, state) = _rnn_step(
						time=time,
						sequence_length=sequence_length,
						min_sequence_length=min_sequence_length,
						max_sequence_length=max_sequence_length,
						zero_output=zero_output,
						state=state,
						call_cell=call_cell,
						state_size=cell.state_size)
			else:
				(output, state) = call_cell()

			outputs.append(output)

		return (outputs, state)

#--------------------------------------------------------------------------------
# rnn_cell.py imports
#--------------------------------------------------------------------------------

class RNNCell(object):
	"""Abstract object representing an RNN cell.

	The definition of cell in this package differs from the definition used in the
	literature. In the literature, cell refers to an object with a single scalar
	output. The definition in this package refers to a horizontal array of such
	units.

	An RNN cell, in the most abstract setting, is anything that has
	a state and performs some operation that takes a matrix of inputs.
	This operation results in an output matrix with `self.output_size` columns.
	If `self.state_size` is an integer, this operation also results in a new
	state matrix with `self.state_size` columns.  If `self.state_size` is a
	tuple of integers, then it results in a tuple of `len(state_size)` state
	matrices, each with a column size corresponding to values in `state_size`.

	This module provides a number of basic commonly used RNN cells, such as
	LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
	of operators that allow add dropouts, projections, or embeddings for inputs.
	Constructing multi-layer cells is supported by the class `MultiRNNCell`,
	or by calling the `rnn` ops several times. Every `RNNCell` must have the
	properties below and and implement `__call__` with the following signature.
	"""

	def __call__(self, inputs, state, scope=None):
		"""Run this RNN cell on inputs, starting from the given state.

		Args:
			inputs: `2-D` tensor with shape `[batch_size x input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
				with shape `[batch_size x self.state_size]`.  Otherwise, if
				`self.state_size` is a tuple of integers, this should be a tuple
				with shapes `[batch_size x s] for s in self.state_size`.
			scope: VariableScope for the created subgraph; defaults to class name.

		Returns:
			A pair containing:

			- Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
			- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		"""
		raise NotImplementedError("Abstract method")

	@property
	def state_size(self):
		"""size(s) of state(s) used by this cell.

		It can be represented by an Integer, a TensorShape or a tuple of Integers
		or TensorShapes.
		"""
		raise NotImplementedError("Abstract method")

	@property
	def output_size(self):
		"""Integer or TensorShape: size of outputs produced by this cell."""
		raise NotImplementedError("Abstract method")

	def zero_state(self, batch_size, dtype):
		"""Return zero-filled state tensor(s).

		Args:
			batch_size: int, float, or unit Tensor representing the batch size.
			dtype: the data type to use for the state.

		Returns:
			If `state_size` is an int or TensorShape, then the return value is a
			`N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

			If `state_size` is a nested list or tuple, then the return value is
			a nested list or tuple (of the same structure) of `2-D` tensors with
		the shapes `[batch_size x s]` for each s in `state_size`.
		"""
		state_size = self.state_size
		if nest.is_sequence(state_size):
			state_size_flat = nest.flatten(state_size)
			zeros_flat = [
					array_ops.zeros(
							array_ops.pack(_state_size_with_prefix(s, prefix=[batch_size])),
							dtype=dtype)
					for s in state_size_flat]
			for s, z in zip(state_size_flat, zeros_flat):
				z.set_shape(_state_size_with_prefix(s, prefix=[None]))
			zeros = nest.pack_sequence_as(structure=state_size,
																		flat_sequence=zeros_flat)
		else:
			zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
			zeros = array_ops.zeros(array_ops.pack(zeros_size), dtype=dtype)
			zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

		return zeros





_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
	"""Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

	Stores two elements: `(c, h)`, in that order.

	Only used when `state_is_tuple=True`.
	"""
	__slots__ = ()

	@property
	def dtype(self):
		(c, h) = self
		if not c.dtype == h.dtype:
			raise TypeError("Inconsistent internal state: %s vs %s" %
											(str(c.dtype), str(h.dtype)))
		return c.dtype




class BasicLSTMCell(RNNCell):
	"""Basic LSTM recurrent network cell.

	The implementation is based on: http://arxiv.org/abs/1409.2329.

	We add forget_bias (default: 1) to the biases of the forget gate in order to
	reduce the scale of forgetting in the beginning of the training.

	It does not allow cell clipping, a projection layer, and does not
	use peep-hole connections: it is the basic baseline.

	For advanced models, please use the full LSTMCell that follows.
	"""

	def __init__(self, num_units, forget_bias=1.0, input_size=None,
							 state_is_tuple=True, activation=tanh):
		"""Initialize the basic LSTM cell.

		Args:
			num_units: int, The number of units in the LSTM cell.
			forget_bias: float, The bias added to forget gates (see above).
			input_size: Deprecated and unused.
			state_is_tuple: If True, accepted and returned states are 2-tuples of
				the `c_state` and `m_state`.  If False, they are concatenated
				along the column axis.  The latter behavior will soon be deprecated.
			activation: Activation function of the inner states.
		"""
		if not state_is_tuple:
			logging.warn("%s: Using a concatenated state is slower and will soon be "
									 "deprecated.  Use state_is_tuple=True.", self)
		if input_size is not None:
			logging.warn("%s: The input_size parameter is deprecated.", self)
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._state_is_tuple = state_is_tuple
		self._activation = activation

	@property
	def state_size(self):
		return (LSTMStateTuple(self._num_units, self._num_units)
						if self._state_is_tuple else 2 * self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		"""Long short-term memory cell (LSTM)."""
		with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
			# Parameters of gates are concatenated into one multiply for efficiency.
			if self._state_is_tuple:
				c, h = state
			else:
				c, h = array_ops.split(1, 2, state)
			concat = _linear([inputs, h], 4 * self._num_units, True)

			# i = input_gate, j = new_input, f = forget_gate, o = output_gate
			global i, j, f, o
			i, j, f, o = array_ops.split(1, 4, concat)



# Would like to acces the i, j, f and o gate weights, trying sess and .eval() for now


			'''
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				ii = i.eval()
				print(ii)
			'''




			new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
							 self._activation(j))
			new_h = self._activation(new_c) * sigmoid(o)

			if self._state_is_tuple:
				new_state = LSTMStateTuple(new_c, new_h)
			else:
				new_state = array_ops.concat(1, [new_c, new_h])
			return new_h, new_state

def _get_concat_variable(name, shape, dtype, num_shards):
	"""Get a sharded variable concatenated into one tensor."""
	sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
	if len(sharded_variable) == 1:
		return sharded_variable[0]

	concat_name = name + "/concat"
	concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
	for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
		if value.name == concat_full_name:
			return value

	concat_variable = array_ops.concat(0, sharded_variable, name=concat_name)
	ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
												concat_variable)
	return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
	"""Get a list of sharded variables with the given dtype."""
	if num_shards > shape[0]:
		raise ValueError("Too many shards: shape=%s, num_shards=%d" %
										 (shape, num_shards))
	unit_shard_size = int(math.floor(shape[0] / num_shards))
	remaining_rows = shape[0] - unit_shard_size * num_shards

	shards = []
	for i in range(num_shards):
		current_size = unit_shard_size
		if i < remaining_rows:
			current_size += 1
		shards.append(vs.get_variable(name + "_%d" % i, [current_size] + shape[1:],
																	dtype=dtype))
	return shards



def _linear(args, output_size, bias, bias_start=0.0, scope=None):
	"""Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

	Args:
		args: a 2D Tensor or a list of 2D, batch x n, Tensors.
		output_size: int, second dimension of W[i].
		bias: boolean, whether to add a bias term or not.
		bias_start: starting value to initialize the bias; 0 by default.
		scope: VariableScope for the created subgraph; defaults to "Linear".

	Returns:
		A 2D Tensor with shape [batch x output_size] equal to
		sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

	Raises:
		ValueError: if some of the arguments has unspecified or wrong shape.
	"""
	if args is None or (nest.is_sequence(args) and not args):
		raise ValueError("`args` must be specified")
	if not nest.is_sequence(args):
		args = [args]

	# Calculate the total size of arguments on dimension 1.
	total_arg_size = 0
	shapes = [a.get_shape().as_list() for a in args]
	for shape in shapes:
		if len(shape) != 2:
			raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
		if not shape[1]:
			raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
		else:
			total_arg_size += shape[1]

	dtype = [a.dtype for a in args][0]

	# Now the computation.
	with vs.variable_scope(scope or "Linear"):
		matrix = vs.get_variable(
				"Matrix", [total_arg_size, output_size], dtype=dtype)
		if len(args) == 1:
			res = math_ops.matmul(args[0], matrix)
		else:
			res = math_ops.matmul(array_ops.concat(1, args), matrix)
		if not bias:
			return res
		bias_term = vs.get_variable(
				"Bias", [output_size],
				dtype=dtype,
				initializer=init_ops.constant_initializer(
						bias_start, dtype=dtype))
	return res + bias_term

#--------------------------------------------------------------------------------
#LSTM RNN Temporal on MNIST handwritten digit recog.
#--------------------------------------------------------------------------------

def recurrent_neural_network(x):
	global layer
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(0, n_chunks, x)

	###lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	lstm_cell = BasicLSTMCell(rnn_size,state_is_tuple=True)

	outputs, states = rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
	



	with tf.Session() as sess:
						sess.run(tf.global_variables_initializer())
						#i = lstm_cell.printer()
						#phony = tf.Print(i, [i])
						weights_ = layer['weights'].eval()
						biases_ = layer['biases'].eval()
						print('Initial Weights: ',weights_)
						print('Initial Biases: ', biases_)

	return output
	
	
def train_neural_network(x):

		prediction = recurrent_neural_network(x)
		cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
		optimizer = tf.train.AdamOptimizer().minimize(cost)


		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
								
			for epoch in range(hm_epochs):
				epoch_loss = 0
				for _ in range(int(mnist.train.num_examples/batch_size)):
					epoch_x, epoch_y = mnist.train.next_batch(batch_size)
					epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

					_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
					epoch_loss += c

				print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			weights___ = layer['weights'].eval()
			print('Updated Weights: ',weights___)
			biases___ = layer['biases'].eval()
			print('Updated Biases: ', biases___)

			#print(i.eval())
			#print('\nPrinting gate weights: ', i)
			#print(i.eval())

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))
train_neural_network(x)
