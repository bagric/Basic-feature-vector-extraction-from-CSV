import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
f1 = open('output','r')
f2 = open('test.txt','r')
f3 = open('konf_matrix.txt','w')
test_result = [int(s.strip()) for s in f1.readlines()]
test_result = np.array(test_result)
np.savetxt('test_res_mat.txt', test_result, delimiter=',', fmt='%d')
test_set = [int(s.split(' ', 1)[0]) for s in f2.readlines()]
test_set = np.array(test_set)
np.savetxt('test_set_mat.txt', test_set, delimiter=',', fmt='%d')
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -99999]
conf_m = str(confusion_matrix(test_set, test_result, labels))
f3.write(conf_m) 