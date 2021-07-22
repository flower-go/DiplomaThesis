import os
import numpy as np
from sklearn.metrics import f1_score

def com_f1(precision, recall):
 return 2 * (precision * recall) / (precision + recall)


directory = "./mtrs"
for filename in os.listdir(directory):
 with open(directory + "/" + filename, 'r') as file:
  data = file.read()
  if len(data) > 0:
    print(filename)
    d = list(map(int, data.split(',')))
 if len(data) > 0:
  m = np.array(d, dtype=np.int32)
  m = np.reshape(m, (3,3))
  f_list = 0
  for i in range(3):
    precision = m[i,i] / np.sum(m[i,:])
    recall = m[i,i]/ np.sum(m[:,i])
    f1 = com_f1(precision, recall)
    f_list += f1
 
   f_macro = f_list/3
   print(f_macro)      	