import glob
import traceback
import os
import shutil
import sys
import torch

def logger(*args):
    print(*args)
    sys.stdout.flush()

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #tf.set_random_seed(seed)


class space (object) : 
    """simple test the namespace of the python
    """

    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self , key , value) : 
        self.namespace[key] = value

    def __getattr__(self , key) : 
        return self.namespace[key]

class fake_space(object):
    """
        return every thing as None
    """
    def __init__ (self) : 
        self.namespace = {}

    def __setitem__(self,  key , value) :
        self.namespace[key] = value

    def __getattr__(self, key):
        return self.namespace[key] if key in self.namespace else None


class Hasher(object) : 
    def __init__(self , li=None) : 
        self.tr = {}
        self.inv = {}
        self.counter = {} # 记录单词出现的频率
        if li != None : 
            self.feed(li)

    def feed(self , li) : 
        assert( isinstance(li, list) )
        cnt = 0
        for name in li : 
            if name not in self.tr : 
                self.tr[name] = cnt 
                self.inv[cnt] = name
                self.counter[name] = 0
                cnt += 1
            self.counter[name] += 1

    def name2id(self , name) : 
        return self.tr[name]

    def id2name(self , idx) : 
        return self.inv[idx]

    def size(self):
        assert(len(self.tr) == len(self.inv))
        return len(self.tr)

    def testcase() : 
        h = Hasher(['name' , 'xk' , 'wt' , 'xk'])
        assert(h.tran('xk') == 1)
        assert(h.tran('name') == 0)
        assert(h.tran('wt') == 2)
        assert(h.invt(2)=='wt')

    def __len__ (self):
        return self.size()

##################################
#
#
##################################
class DelayArgProcessor:
    """
       这个函数用来延迟处理参数输入和输出，参数一个一个传递，只要搜集了n个参数，就开始调用process函数
       process 函数包括两步，第一个是检测参数，
    """
    def __init__(self, n, process_func, post_process=None, assert_func=None):
        self.n = n 
        self.args = []
        self._func_ = process_func
        self._post_ = post_process
        self._check_ = assert_func

    def process(self, input_args) : 
        self.args.append(input_args)
        if len(self.args) == self.n : 
            import pdb
            #pdb.set_trace()
            if self._check_ :
                try : 
                    res = [ self._check_(i) for i in self.args ]
                    assert (sum(res) == 0)
                except AssertionError as e : 
                    print ('[ERROR] input args not compatiable')

            self._func_(*self.args)
            self.args = self._post_(self.args) if self._post_ else []

def tensor_compare_delay_factory():
    def func(a, b):
        print('[Compare]')
        print('\t', a)
        print('\t', b)
    return DelayArgProcessor(2, func)

class Once:
    def __init__(self, func):
        self.init = True
        self.func = func

    def process(self, *args):
        if self.init:
            self.func(*args)
        self.init = False

class Every:
    def __init__(self, func, step):
        self.step_origin = step
        self.step = step
        self.func = func

    def process(self, *args):
        self.step = self.step - 1
        if self.step == 0:
            self.func(*args)
            self.step = self.step_origin

class DataUnit(object):
    def __init__(self, name, process_func, save_func=None, load_func=None, save_path=None, load_path=None, debug=None, **args):
        if load_path == None and save_path != None:
            load_path = save_path
        if load_path != None and save_path == None:
            save_path = load_path

        self.name = name
        self.proc = process_func
        self.save = save_func
        self.load = load_func
        self.save_path = save_path
        self.load_path = load_path
        self.args = args
        self.debug = debug

        #if self.load_path and not os.path.exists(self.load_path) : os.mkdir(self.load_path)
        #if self.save_path and not os.path.exists(self.save_path) : os.mkdir(self.save_path)
    
    def process(self):
        output = None
        if self.load and self.load_path:
            try : 
                output = self.load(self.args, self.load_path)
                return output
            except : 
                print ('[ERROR: DataUnit %s]: load failed, reprocess and resave' % self.name)
                if self.debug: traceback.print_exc()
        print ('[DEBUG: DataUnit %s]: start process' % self.name)
        output = self.proc(self.args)
        if self.save and self.save_path:
            print ('[DEBUG: DataUnit %s]: start save' % self.name)
            self.save(self.args, output, self.save_path)
        return output 

class xk_Maximize(object):
    def __init__(self, keys, init):
        assert (isinstance(keys, list))
        assert (isinstance(keys[0], str))
        self.keys = keys
        self.init = init
        self.reset()

    def _max_(self, a, b, keys):
        """ if a > b : return true"""
        res = True
        for key in keys:
            if a[key] > b[key] : return True
            if a[key] < b[key] : return False
        return False

    def reset(self):
        self._best_ = { key:self.init[i] for i,key in enumerate(self.keys) }
        self._start_ = False
        
    def process(self, inp):
        """ inp will be compare and find the max inp"""
        self._start_ = True
        if self._max_(inp, self._best_, self.keys): 
            self._best_ = inp.copy()
        return None

    def getdata(self):
        return self._best_.copy()

import sys
def on_send_uicmd(li):
    li.insert(0, 'uicmd')
    sys.stderr.write('\r' + ' '.join(li) + '\r')
    sys.stderr.flush()

def stack_data_from_batch(batched_inputs, key, dtype, device='cuda'):
    """
        key     [string]         the key name of gathered data
        device  [string]         'cpu' | 'cuda'   the place of tensor
        dtype   [torch.type]     torch.float32 | torch.float64 ....  [tensor type](https://pytorch.org/docs/stable/tensors.html)
    """
    assert (len(batched_inputs) > 0)
    assert (key in batched_inputs[0]) 
    tensors = [torch.as_tensor(b[key]) for b in batched_inputs]
    return torch.stack(tensors, dim=0).to(device).to(dtype)

if __name__ == '__main__':
    print(' ======================= DataUnit =========================')
    def test_proc(m):
        print ('yes')
        d = {'a':1}
        m.d = d
        return d

    def test_save(m, p):
        import pickle
        pickle.dump(m.d, open(p+'yes.pkl', 'wb'))

    def test_load(m, p):
        import pickle
        return pickle.load(open(p+'yes.pkl', 'rb'))

    test = DataUnit('test', test_proc, test_save, test_load, "./cache/test/", None, 'yes')
    data = test.process()
    assert(data['a'] == 1)
