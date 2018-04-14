
"""
"""

import pandas as pd
from udumbara import IV, WOE, PSI, ROC, AUC, KS, KS_from_suit
from udumbara import IV_from_suit, WOE_from_suit
from discrete import *

result = {}
df = pd.read_csv('test.csv')
x_201609 = df[df['date_key'] == 201609].iloc[:,5:]
y_201609 = df[df['date_key'] == 201609]['gbie']
x_201611 = df[df['date_key'] == 201611].iloc[:,5:]
y_201611 = df[df['date_key'] == 201611]['gbie']

for cache in Discreter_2suits(x_201609, y_201609, 
					x_201611,y_201611, n_cuts=5, missing_value=-99999):
	(expect_suit, actual_suit, label, group_info) = cache
	result[label] = {}
	result[label]['group'] = group_info
	result[label]['PSI'] = PSI(expect_suit, actual_suit)

for cache in Discreter(x_201609, y_201609, n_cuts=5, missing_value=-99999):
	(suit, label, _) = cache
	result[label]['IV'] = IV_from_suit(suit, pos=1, neg=0)
	result[label]['KS'] = KS_from_suit(suit, pos=1, neg=0, verbose=True)
	
import json
with open('result.json', 'w') as f:
	json.dump(result, f)