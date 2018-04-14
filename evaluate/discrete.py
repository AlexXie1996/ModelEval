
"""
"""
import pandas as pd
import numpy as np

def _Binning(data, max, min, n_cuts):
	"""
	"""
	gap = (max-min) / n_cuts
	# group information
	gaps = [str(int(min + i*gap)) for i in range(n_cuts+1)]
	gaps[0] = '-inf'
	gaps[-1] = 'inf'
	group_info = [gaps[i]+'-'+gaps[i+1] for i in range(n_cuts)]
	
	# cal df['group']
	group = pd.Series((data.iloc[:,0]-min) // gap)
	group[group <= 0] = 0
	group[group >= n_cuts-1] = n_cuts-1
	
	# groupby
	data['group'] = group
	for i in range(n_cuts):
		yield data[data['group'] == i], group_info[i]
	
	return 'finish'
	
def Discreter(df, y, n_cuts=5, missing_value=np.nan):
	"""
	"""
	for var in df:	
		# replace missing value with -1
		data = pd.concat([df[var], y], axis=1).replace(missing_value, -1)
		max, min = data.iloc[:,0].max(), data.iloc[:,0].min()
		
		suit = []
		group_info = []
		for v, g in _Binning(data, max, min, n_cuts):
			if not v.empty:
				del v['group']
				suit.append(v.as_matrix())
				group_info.append(g)
		yield (suit, var, group_info)
	
	return 'finish'
		
		
def Discreter_2suits(df1, y1, df2, y2, n_cuts=5, missing_value=np.nan):
	"""
	"""
	for var in df1:
		# replace missing value with -1
		data1 = pd.concat([df1[var], y1], axis=1).replace(missing_value, -1)
		data2 = pd.concat([df2[var], y2], axis=1).replace(missing_value, -1)

		suit1, suit2 = [], []
		group_info = []
		max, min = data1.iloc[:,0].max(), data1.iloc[:,0].min()
		# binning by max and min of df1

		bin_generator = _Binning(data2, max, min, n_cuts)
		for v1, g in _Binning(data1, max, min, n_cuts):
			v2, _ = next(bin_generator)
			if not v1.empty or not v2.empty:
				del v1['group']
				del v2['group']
				suit1.append(v1.as_matrix())
				suit2.append(v2.as_matrix())
				group_info.append(g)
	
		yield (suit1, suit2, var, group_info)
	
	return 'finish'
		