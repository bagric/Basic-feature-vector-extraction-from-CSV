import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
f1 = open('output','r')
f2 = open('test.txt','r')
f3 = open('konf_matrix.txt','w')
test_result = [int(s.strip()) for s in f1.readlines()]
test_result = np.array(test_result)
test_set = [int(s.split(' ', 1)[0]) for s in f2.readlines()]
test_set = np.array(test_set)
conf_m = str(confusion_matrix(test_set, test_result))
f3.write(conf_m)
