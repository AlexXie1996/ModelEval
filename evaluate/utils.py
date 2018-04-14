
"""
"""

import numpy as np
import copy

def is_array_like(x):
	"""
	"""
	if hasattr(x, '__len__') or hasattr(x, 'shape') \
			or hasattr(x, '__array__'):
		return True
	else:
		return False

def unpacking_suit(suit):
	"""
	"""
	return [sample for box in suit for sample in box]

def unpacking_suit_array(suit):
	"""
	"""	
	return np.array(unpacking_suit(suit))

def num_dims(x):
	"""
	"""
	if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
		if hasattr(x, '__array__'):
			x = np.asarray(x)
		else:
			raise TypeError('Expected sequence-like '
					'got %s' % type(x))
	
	if hasattr(x, 'shape'):
		if len(x.shape) == 0:
			raise TypeError('Singleton array can not be '
					'considered a valid collection')
		else:
			return x.shape[0]
	else:
		return len(x)

def check_consistent_lenght(X, dims=None):
	"""
	"""
	lengths = [num_dims(i) for i in X if i is not None]
	uniques = set(lengths)
	if len(uniques) == 0:
		raise ValueError('Null array can not be considered valid')
	if len(uniques) > 1:
		raise ValueError('Found input valuables with inconsistent '
				'dims')
	if dims != None:
		all_dims = uniques.pop() 
		if all_dims != dims:
			raise TypeError('Found input valuables with different'
					 'dim %d and %d' % (dims, all_dims))

def _filter_elem(sequence, elem):
	"""
	"""
	return list(filter(lambda x: False if x==elem else True, sequence))

def _binary_check_list(x):
	"""
	"""
	rest = _filter_elem(x, x[0])
	if len(rest) == 0:
		return
	rest = _filter_elem(rest, rest[0])
	if len(rest) > 0:
		raise ValueError('Expected binary sequence, got multiple')

def binary_check(x):
	"""
	"""
	return _binary_check_list(x)
	
def check_suit(suit):
	"""
	"""
	if suit is None:
		raise TypeError('Found input is None')
	if is_array_like(suit) == False:
		raise TypeError('Suit expected array-like '
					'got %s' % type(suit))
	unpacked_suit = unpacking_suit(suit)
	check_consistent_lenght(unpacked_suit, 2)

	y = [samples[1] for samples in unpacked_suit]
	_binary_check_list(y)

def check_targets(x, y):
	"""
	"""
	if x is None or y is None:
		raise TypeError('Found input is None')
	if is_array_like(x) == False or is_array_like(y) == False:
		raise TypeError('Suit expected array-like input')

	x_dim = num_dims(x)
	y_dim = num_dims(y)
	if x_dim != y_dim:
		raise TypeError('Found input valuables with different '
					 'dim %d and %d' % (x_dim, y_dim))

def distributor(collections):
	"""
	"""
	for box in collections:
		yield np.array(box)
	return 'finish'

def _numerical_func(pos, neg):
	def f(x):
		if x == pos:
			return 1
		elif x == neg:
			return 0
		else:
			raise ValueError('y can not be normed by %r and %r' % (pos, neg))
	return f
			
def norm_numerization(y, pos, neg):
	"""
	"""
	return np.array(list(map(_numerical_func(pos, neg), y)))
	
def suit_numerization(suit, pos, neg):
	"""
	"""
	suit_copy = copy.deepcopy(suit)
	for box in suit_copy:
		box[:, 1] = norm_numerization(box[:,1], pos, neg)
	return np.array(suit_copy)

