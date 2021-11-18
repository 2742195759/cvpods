#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   npy2txt.py
@Time               :   2021-08-21
@Author             :   Kun Xiong
@Contact            :   xk18@mails.tsinghua.edu.cn
@Last Modified by   :   2021-11-07
@Last Modified time :   2021-11-07

Convert the npy matrix to txt matrix
'''


import numpy as np
import sys
import os.path as osp
assert len(sys.argv) >= 2, "please input the args.: filename int|float|double"
name = sys.argv[1]
dtype = sys.argv[2]
tt = np.load("".join([name, '.npy'])).astype(dtype)
tt[tt>5] = 5
np.savetxt("".join([name, '.ascii']), tt, fmt='%1d')
