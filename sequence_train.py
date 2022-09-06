import torch
from torch import optim
import torch.nn.functional as F
import generate_data
from utils import to_var
from pointer_network import PointerNetwork
import my_ext
import ctypes

# Run in console the following: Nvidia-smi


load_from_file = False
total_size = 1000
weight_size = 256
emb_size = 32 # embedding size of 32
batch_size = 250 # Smaller values are slower but converge faster {or: 250}
n_epochs = 500
input_seq_len = 8
input, targets = generate_data.make_seq_data(total_size, input_seq_len)
inp_size = input_seq_len
print('Data sequence was generated.')
print('Input size:', inp_size)
# my_ext.save_route_repr(input, 'gdata', 'input')
# my_ext.save_route_repr(targets, 'gdata', 'targets')

# Convert to torch tensors
input = to_var(torch.LongTensor(input))     # (N, L)
targets = to_var(torch.LongTensor(targets)) # (N, L)

data_split = (int)(total_size * 0.9)
train_X = input[:data_split]
train_Y = targets[:data_split]
test_X = input[data_split:]
test_Y = targets[data_split:]


print('Train X size:', train_X.size())
print('Test X size:', test_X.size())

model = PointerNetwork(inp_size, emb_size, weight_size, input_seq_len)
# print(list(model.parameters()))
print('Model information:')
print(model)
print('-' * 80)

if torch.cuda.is_available():
    my_ext.print_cuda_info()
    model.cuda()
else:
    print('CUDA is not available! Training on CPU...')

print('Training model initiated...')

# from pointer_network import PointerNetwork
def train_model(model, X, Y, batch_size, n_epochs):
    """
    `train_model` function accepts as inputs a model, training data, batch size, and number of epochs.

    It then iterates through the data in batches of size batch_size for n_epochs number of times.
    The loss is calculated by taking the negative log likelihood loss function from pytorch's functional module.

    :param model: Pass the model to be trained
    :param X: Pass the input data to the model
    :param Y: Calculate the accuracy of the model
    :param batch_size: Determine how many sequences to process at once
    :param n_epochs: Determine how many times the model should be trained
    :return: The trained model
    """
    if load_from_file:
        model.load_state_dict(torch.load('entire_model.pt'))
        model.eval()
    else:
        model.train()
        optimizer = optim.Adam(model.parameters())
    N = X.size(0)
    L = X.size(1)
    # M = Y.size(1)
    for epoch in range(n_epochs + 1):
        # for i in range(len(train_batches))
        for i in range(0, N-batch_size, batch_size):
            x = X[i:i+batch_size] # (bs, L)
            y = Y[i:i+batch_size] # (bs, M)

            probs = model(x) # (bs, M, L)
            outputs = probs.view(-1, L) # (bs*M, L)
            # outputs = probs.view(L, -1).t().contiguous() # (bs*M, L)
            y = y.view(-1) # (bs*M)
            loss = F.nll_loss(outputs, y)
            # loss = F.cross_entropy(outputs, y)

            if not load_from_file:
                optimizer.zero_grad()
                loss.backward() # cudnn RNN backward can only be called in training mode
                optimizer.step()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            test(model, X, Y)
            #  Break if accuracy is good enough
            if loss.item() < 0.01:
                print('Loss is less than 0.01. Breaking and saving...')
                # Save model via state_dict
                torch.save(model.state_dict(), 'entire_model.pt')
                break




def test(model, X, Y):
    probs = model(X) # (bs, M, L)
    _v, indices = torch.max(probs, 2) # (bs, M)
    # show test examples
    # for i in range(len(indices)):
    #     print('-----')
    #     print('test', [v for v in X[i].data])
    #     print('label', [v for v in Y[i].data])
    #     print('pred', [v for v in indices[i].data])
    #     if torch.equal(Y[i].data, indices[i].data):
    #         print('eq')
    #     if i>20: break
    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, Y)])
    print('Acc: {:.5f}% ({}/{})'.format(correct_count/len(X)*100, correct_count, len(X)))




train_model(model, train_X, train_Y, batch_size, n_epochs)
print('----Test result---')
test(model, test_X, test_Y)

