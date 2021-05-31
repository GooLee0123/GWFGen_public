import os
import math
import logging

import numpy as np
import pandas as pd
import torch

from operator import itemgetter

class DatPrep(object):

    """
        Class for pre- and post-processing of gravitational wave data.

        Inputs: path, header, sfactor
            - **path** (str): Source directory of gravitational wave data 
            - **header** (int, optional): Existence of header on the source file, (default: None)
            - **sfactor** (float, optional): Scale factor for input data, (default: 1e22)
    """

    def __init__(self, fdir):
        self.logger = logging.getLogger(__name__)
        self.fdir = fdir

    def load(self, path, header=None):
        self.logger.info("Load datasets from {} files". \
            format(path))
        self.path = path
        self.data = pd.read_csv(path, header=header).values

        return self.data

    def tolist(self):
        self.logger.info("Transform data from pandas object to python list")
        self.data = list(self.data)

        return self.data

    def tofloat(self, blank=True):
        self.logger.info("Float data elements")
        for i in range(len(self.data)):
            if blank:
                self.data[i] = map(float, self.data[i][0].split(' ')[1:])
            else:
                self.data[i] = map(float, self.data[i][0].split(' '))
        return self.data

    def nonzero(self):
        self.logger.info("Get rid of zero value components")

        for j in range(len(self.data)):
            self.data[j] = self.data[j][:1+np.nonzero(np.array(self.data[j]))[0][-1]]

        return self.data

    def scale(self, sfactor=1e22):
        for i in range(len(self.data)):
            self.data[i] = (pd.Series(self.data[i]*sfactor)).tolist()

        return self.data

    def normalize(self, strain='hp'):
        self.logger.info("Normalize data elements with the maximum value of data")

        fmax_amp = os.path.join(self.fdir, 'maximum_amplitude')

        self.logger.info("Read maximum amplitude from {}".format(fmax_amp))

        if strain is 'hp':
            idx = 0
        elif strain is 'hc':
            idx = 1
        else:
            raise ValueError("strain must be 'hp' or 'hc'")

        max_amp = np.genfromtxt(fmax_amp)[idx]

        for k in range(len(self.data)):
            self.data[k] = (pd.Series(self.data[k])/max_amp).tolist()

        return self.data

    def sort_by_len(self, key, data_si=None):
        self.logger.info("Sort data by length")

        if key == 'input':
            data_si = sorted(range(len(self.data)), key=lambda k: -len(self.data[k]))

            sort_function = lambda x,y: cmp(len(y), len(x))

            self.data.sort(sort_function)

            return self.data, data_si
        else:
            self.data = list(np.array(self.data)[data_si])

            return self.data

    def get_input_len(self, isize):
        self.logger.info("Get input length")

        ilens = []
        rlens = []

        for dat in self.data:
            ilens.append(int(math.ceil(len(dat)/float(isize))))
            rlens.append(len(dat))

        return ilens, rlens

    def load_split(self, train, val, header=None):
        self.logger.info("Load datasets from {}, {} files". \
            format(train, val))

        self.train = pd.read_csv(train, header=header).values
        self.val = pd.read_csv(val, header=header).values

        return self.train, self.val

    def tolist_split(self):
        self.logger.info("Transform data from pandas object to python list")

        self.train = list(self.train)
        self.val = list(self.val)

        return self.train, self.val

    def tofloat_split(self):
        self.logger.info("Float data elements")

        for i in range(len(self.train)):
            self.train[i] = map(float, self.train[i][0].split(' ')[1:])
        for j in range(len(self.val)):
            self.val[j] = map(float, self.val[j][0].split(' ')[1:])

        return self.train, self.val

    def nonzero_split(self):
        self.logger.info("Get rid of zero value components")

        for i in range(len(self.train)):
            self.train[i] = self.train[i][:1+np.nonzero(np.array(self.train[i]))[0][-1]]
        for j in range(len(self.val)):
            self.val[j] = self.val[j][:1+np.nonzero(np.array(self.val[j]))[0][-1]]

        return self.train, self.val

    def scale_split(self, sfactor=1e22):
        self.logger.info("Scale data elements with scale factor %s" % sfactor)

        for i in range(len(self.train)):
            self.train[i] = (pd.Series(self.train[i])*sfactor).tolist()
        for j in range(len(self.val)):
            self.val[j] = (pd.Series(self.val[j])*sfactor).tolist()

        return self.train, self.val

    def normalize_split(self, strain='hp'):
        self.logger.info("Normalize data elements with the maximum value of data")

        fmax_amp = os.path.join(self.fdir, 'maximum_amplitude')

        self.logger.info("Read maximum amplitude from {}".format(fmax_amp))

        if strain is 'hp':
            idx = 0
        elif strain is 'hc':
            idx = 1
        else:
            raise ValueError("strain must be 'hp' or 'hc'")

        max_amp = np.genfromtxt(fmax_amp)[idx]

        for i in range(len(self.train)):
            self.train[i] = (pd.Series(self.train[i])/max_amp).tolist()
        for j in range(len(self.val)):
            self.val[j] = (pd.Series(self.val[j])/max_amp).tolist()

        return self.train, self.val

    def sort_by_len_split(self, key, train_si=None, val_si=None):
        self.logger.info("Sort data by length")

        if key == 'input':
            train_si = sorted(range(len(self.train)), key=lambda k: -len(self.train[k]))
            val_si = sorted(range(len(self.val)), key=lambda k: -len(self.val[k]))

            sort_function = lambda x,y: cmp(len(y), len(x))

            self.train.sort(sort_function)
            self.val.sort(sort_function)

            return self.train, self.val, train_si, val_si
        else:
            self.train = list(np.array(self.train)[train_si])
            self.val = list(np.array(self.val)[val_si])

            return self.train, self.val

    def get_input_len_split(self, isize):
        self.logger.info("Get input length")

        data = [self.train, self.val]
        ilens = [[], []]
        rlens = [[], []]

        for i, dat in enumerate(data):
            for row in dat:
                ilens[i].append(int(math.ceil(len(row)/float(isize))))
                rlens[i].append(len(row))
        return ilens[0], ilens[1], rlens[0], rlens[1]

    def get_iter_arr(self, dlen, batch_size, shuffle=False):
        if shuffle:
            iter_arr = np.arange(0, len(dlen), batch_size)
            np.random.shuffle(iter_arr)
        else:
            iter_arr = range(0, len(dlen), batch_size)

        return iter_arr

    def pad_batch_by_order(self, data, dlen, rdlen, batch_size, isize, dtype,
                            iter_arr, key, train=False, repeatable=True, inp=None):
        self.logger.info("Get batch input generator by order")

        cuda = torch.cuda.is_available()

        # if cuda:
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     torch.set_default_tensor_type('torch.FloatTensor')

        def _roundup(x, th):
            return int(math.ceil(x/float(th))*th)

        def _get_maxlen(data, isize):
            maxdat = max(enumerate(data), key=lambda x:len(x[1]))[1]
            maxlen = _roundup(len(maxdat), isize)
            return maxlen

        if repeatable:
            while True:
                for i in iter_arr:
                    mlength = _get_maxlen(data[i:i+batch_size], isize)
                    batch = []
                    for j in data[i:i+batch_size]:
                        if key == 'target':
                            batch.append([0.]*isize+j+[0.]*(mlength-len(j)))
                        else:
                            # batch.append(torch.tensor(j+[0.]*(mlength-len(j))))
                            batch.append(torch.tensor(j))
                    if key == 'target':
                        seq_len = int(math.ceil(len(batch[0])/float(isize)))
                        if cuda:
                            yield torch.tensor(batch, dtype=dtype).view(-1, seq_len, isize).cuda(), \
                                torch.tensor(dlen[i:i+batch_size], dtype=torch.long).cuda(), \
                                torch.tensor(rdlen[i:i+batch_size], dtype=torch.long).cuda()
                        else:
                            yield torch.tensor(batch, dtype=dtype).view(-1, seq_len, isize), \
                                torch.tensor(dlen[i:i+batch_size], dtype=torch.long)
                    else:
                        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
                        padded = padded.view(padded.size(0), -1, isize)
                        if cuda:
                            yield torch.nn.utils.rnn.pack_padded_sequence(padded, dlen[i:i+batch_size], batch_first=True).cuda(),\
                                padded,\
                                torch.tensor(dlen[i:i+batch_size], dtype=torch.long).cuda()
                        else:
                            yield torch.nn.utils.rnn.pack_padded_sequence(padded, dlen[i:i+batch_size], batch_first=True), \
                                torch.tensor(dlen[i:i+batch_size], dtype=torch.longdtype)
        else:
            for i in iter_arr:
                mlength = _get_maxlen(data[i:i+batch_size], isize)
                batch = []
                for j in data[i:i+batch_size]:
                    if key =='target':
                        batch.append(j+[0.]*(mlength-len(j)))
                    else:
                        batch.append(torch.tensor(j+[0.]*(mlength-len(j))))
                        # batch.append(torch.tensor(j))
                if key == 'target':
                    seq_len = int(math.ceil(len(batch[0])/float(isize)))
                    if cuda:
                        yield torch.tensor(batch, dtype=dtype).view(-1, seq_len, isize).cuda(), \
                            torch.tensor(dlen[i:i+batch_size], dtype=torch.long).cuda(), \
                            torch.tensor(rdlen[i:i+batch_size], dtype=torch.long).cuda()
                    else:
                        yield torch.tensor(batch, dtype=dtype).view(-1, seq_len), \
                            torch.tensor(dlen[i:i+batch_size], dtype=torch.long)
                else:
                    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
                    #padded = padded.unsqueeze(-1)
                    padded = padded.view(padded.size(0), -1, isize)
                    # lengths = torch.as_tensor(dlen[i:i+batch_size], dtype=torch.int64)
                    # lengths = lengths.cpu()
                    if cuda:
                        yield torch.nn.utils.rnn.pack_padded_sequence(padded, dlen[i:i+batch_size], batch_first=True).cuda(), \
                            torch.tensor(dlen[i:i+batch_size], dtype=torch.long).cuda()
                    else:
                        yield torch.nn.utils.rnn.pack_padded_sequence(padded, dlen[i:i+batch_size], batch_first=True), \
                            torch.tensor(dlen[i:i+batch_size], dtype=torch.long)
