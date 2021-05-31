import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

from networks.seq2seq import EncoderRNN
   
cuda = torch.cuda.is_available() 
class GWDataset(Dataset):
    def __init__(self, data , params=None):
        self.params = params
        self.data = data

    def preprocessing(self):
    	padded = torch.nn.utils.rnn.pad_sequence(self.data, batch_first=True)
		padded = padded.unsqueeze(-1)
		pack = torch.nn.utils.rnn.pack_padded_sequence(padded, dlen[::-1], batch_first=True).cuda
        #normalization
        self.data = pack

    def __getitem__(self, index):
    	
    	return self.data[index]
        # data = self.preprocessing(self.data[index])
        # split = 100
        # # split into input and target
        # # calcuate splitting point
        # input = np.expand_dims(data[index][:split],0)
        # target = np.expand_dims(data[index][split:split+split],0)
        # return torch.from_numpy(input), torch.from_numpy(target)

    def __len__(self):
        return len(self.data)

dlen = np.arange(256, 288, 1)
data = []
for l in dlen[::-1]:
	data.append(torch.tensor(np.random.randn(l)))

gwdataset = GWDataset(data)
train_loader = DataLoader(gwdataset,batch_size=2, shuffle=True,
                              num_workers=4, pin_memory=cuda)

print train_loader

encoder = EncoderRNN(1, 256, num_stacked_layer=4,
					input_dropout_p=0,
					dropout_p=0,
					bidirectional=False,
					rnn_cell='gru')

encoder = torch.nn.DataParallel(encoder, device_ids=[0,1]).cuda()
for epoch in range(0, 100):
        for idx, train_data in enumerate(train_loader):
            input = train_data

            #if cuda:
            #   input,target = input.cuda(), target.cuda()
            print(idx, input.shape)
            #out = model(input)
            #loss = target - out
            #loss.backward()
			encoder(input)