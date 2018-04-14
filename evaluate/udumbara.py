
"""
"""

import numpy as np
from utils import is_array_like, unpacking_suit_array, distributor
from utils import check_suit, check_targets, suit_numerization
from utils import unpacking_suit, num_dims, binary_check, norm_numerization

_EPSILON = .001

def _count_respone(samples):
	"""
	"""
	n_samples, n_dims = samples.shape
	if n_samples == 0:
		raise ValueError('Found no samples')
	
	respone = np.sum(samples[:,-1])
	return respone, n_samples

def _cal_IV_score(box, y_respones, y_samples, y_log_prob):
	"""
	"""
	n_respones, n_samples = _count_respone(box)

	woe = (np.log(n_respones+_EPSILON) - \
			np.log(n_samples-n_respones+_EPSILON)) - y_log_prob
	coefficient = (n_respones+_EPSILON) / (y_respones+_EPSILON) - \
		(n_samples-n_respones+_EPSILON) / (y_samples-y_respones+_EPSILON)

	return coefficient * woe

def IV_from_suit(suit, pos=1, neg=0):
	"""
	"""
	# check input before calculate
	check_suit(suit)

	# init numerical reader for suit
	numerical_suit = suit_numerization(suit, pos, neg)
	unpacked_suit = unpacking_suit_array(numerical_suit)
	suit_reader = distributor(numerical_suit)

	# calculate IV score
	respones, samples = _count_respone(unpacked_suit)
	y_log_prob = np.log(respones+_EPSILON)-np.log(samples-respones+_EPSILON)

	iv = 0
	for box in suit_reader:
		iv += _cal_IV_score(box, respones, samples, y_log_prob)

	return iv

def IV(packet, labels, pos=1, neg=0):
	"""
	params:
		- packet: 5-D array-like
	"""
	check_targets(packet, labels)

	re_dict = {}
	for i, suit in enumerate(packet):
		re_dict[labels[i]] = IV_from_suit(suit, pos=pos, neg=neg)
	return re_dict
	
def _cal_WOE_score(box, y_log_prob):
	"""
	"""
	n_respones, n_samples = _count_respone(box)
	n_log_prob = np.log(n_respones+_EPSILON) - \
					np.log(n_samples-n_respones+_EPSILON)
	return n_log_prob - y_log_prob
	
def WOE_from_suit(suit, pos=1, neg=0):
	"""
	"""
	# check input before calculate
	check_suit(suit)

	# init numerical reader for suit
	numerical_suit = suit_numerization(suit, pos, neg)
	unpacked_suit = unpacking_suit_array(numerical_suit)
	suit_reader = distributor(numerical_suit)

	# calculate IV score
	respones, samples = _count_respone(unpacked_suit)
	y_log_prob = np.log(respones+_EPSILON) - \
					np.log(samples-respones+_EPSILON)

	woe_list = []
	for box in suit_reader:
		woe_list.append(_cal_WOE_score(box, y_log_prob))

	return woe_list
	
def WOE(packet, labels, pos=1, neg=0):
	"""
	"""
	check_targets(packet, labels)
	
	re_dict = {}
	for i, suit in enumerate(packet):
		re_dict[labels[i]] = WOE_from_suit(suit, pos=pos, neg=neg)
					
	return re_dict

def _cal_PSI_score(expect_box, n_expect, actual_box, n_actual):
	"""
	"""
	expect = num_dims(expect_box)
	actual = num_dims(actual_box)
	
	expect_rate = (expect+_EPSILON) / (n_expect+_EPSILON)
	actual_rate = (actual+_EPSILON) / (n_actual+_EPSILON)
	
	return (actual_rate - expect_rate) * \
				(np.log(actual_rate) - np.log(expect_rate))
				
def PSI(expect_suit, actual_suit):
	"""
	"""
	check_targets(expect_suit, actual_suit)
	
	unpacked_expect_suit = unpacking_suit(expect_suit)
	unpacked_actual_suit = unpacking_suit(actual_suit)
	
	n_box = num_dims(expect_suit)
	n_expect = num_dims(unpacked_expect_suit)
	n_actual = num_dims(unpacked_actual_suit)
	
	psi = 0
	for i in range(n_box):
		psi += _cal_PSI_score(expect_suit[i], n_expect,
								actual_suit[i], n_actual)
								
	return psi
	
def _count_goods(x):
	"""
	"""
	n_total = num_dims(x)
	n_goods = np.sum(x)
	n_bads = n_total - n_goods
	
	return n_goods, n_bads

def _get_positive_rate(x):
	"""
	"""
	n_goods, n_bads = _count_goods(x)
	
	tpr, fpr = np.zeros_like(x),np.zeros_like(x)
	acc_tpr, acc_fpr = 0,0
	for index, yi in enumerate(x):
		if yi == 1:
			acc_tpr += 1
		elif yi == 0:
			acc_fpr += 1
		tpr[index] = acc_tpr
		fpr[index] = acc_fpr
	return tpr/n_goods, fpr/n_bads
	
def ROC(y_pred, y_true, pos=1, neg=0):
	"""
	"""
	check_targets(y_pred, y_true)
	binary_check(y_true)
	y_norm = norm_numerization(y_true, pos, neg)

	try:
		sorted_id = np.argsort(-np.array(y_pred))
	except:
		raise TypeError('Found input can not be sorted')
	
	y_norm = y_norm[sorted_id]
	tpr, fpr = _get_positive_rate(y_norm)

	return tpr, fpr

def _cal_auc_score(x):
	"""
	"""
	n_goods, n_bads = _count_goods(x)
	
	acc_tpr, acc_fpr, auc = 0,0,0
	for index, yi in enumerate(x):
		if yi == 1:
			acc_tpr += 1
		elif yi == 0:
			auc += acc_tpr
		
	return auc / (n_goods * n_bads)
	
def AUC(y_pred, y_true, pos=1, neg=0):
	"""
	"""
	check_targets(y_pred, y_true)
	binary_check(y_true)
	y_norm = norm_numerization(y_true, pos, neg)

	try:
		sorted_id = np.argsort(-np.array(y_pred))
	except:
		raise TypeError('Found input can not be sorted')
	
	y_norm = y_norm[sorted_id]
	return _cal_auc_score(y_norm)
		
def KS(y_pred, y_true, pos=1, neg=0):
	"""
	"""
	tpr, fpr = ROC(y_pred, y_true, pos, neg)
	
	diff = tpr - fpr
	return np.max(diff), np.argmax(diff)

def _cal_KS_score(box, n_goods, n_bads, acc_goods, acc_bads):
	"""
	"""
	if n_goods == 0 or n_bads == 0:
		raise ValueError('Found no pos or neg samples')
	respones, samples = _count_respone(box)
	acc_goods += respones
	acc_bads += (samples-respones)
	ks = abs((acc_goods / n_goods) - (acc_bads / n_bads))
	return ks, acc_goods, acc_bads
	
def KS_from_suit(suit, pos=1, neg=0, verbose=False):
	"""
	"""
	# check input before calculate
	check_suit(suit)

	# init numerical reader for suit
	numerical_suit = suit_numerization(suit, pos, neg)
	unpacked_suit = unpacking_suit_array(numerical_suit)
	suit_reader = distributor(numerical_suit)

	# calculate KS score
	respones, samples = _count_respone(unpacked_suit)
					
	ks_list = []
	n_goods, n_bads = respones, (samples-respones)
	acc_goods, acc_bads = 0, 0
	for box in suit_reader:
		ks, acc_goods, acc_bads = _cal_KS_score(box, n_goods, n_bads, 
					acc_goods, acc_bads)
		ks_list.append(ks)
	if verbose:
		return ['group{0}: {1}'.format(i, ks) for i,ks in enumerate(ks_list)]
	return 'group{0}: {1}'.format(str(np.argmax(ks_list)),
										str(np.max(ks_list)))
	