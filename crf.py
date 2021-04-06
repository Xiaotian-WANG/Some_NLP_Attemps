import torch
from TorchCRF import CRF
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
sequence_size = 3
num_labels = 5
mask = torch.ByteTensor([[1, 1, 1]]).to(device) # (batch_size. sequence_size)
labels = torch.LongTensor([[0, 2, 3]]).to(device)  # (batch_size, sequence_size)
hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True).to(device)
crf = CRF(num_labels).to(device)
print(crf.forward(hidden, labels, mask))
print(crf.viterbi_decode(hidden, mask))
print('finished')
