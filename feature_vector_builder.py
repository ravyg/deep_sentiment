#!/usr/bin/env python
# This file does the part to automated annotation.
# -*- coding: utf-8 -*-
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')
count_positive=0
count_negative=0
extra=500
balanced_class_vector = []
positive_list = ['Positive', 'Verypositive']
for line in open('feature_vector.csv','r').readlines():
  if 'Positive' in line:
    balanced_class_vector.append(line.rstrip())
    count_positive=count_positive+1
  else:
    if count_negative <= count_positive+extra:
      count_negative=count_negative+1
      balanced_class_vector.append(line.rstrip())

np.savetxt('feature_vector_balanced.csv', balanced_class_vector, delimiter=",", fmt="%s")
