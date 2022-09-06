import torch
from torch.autograd import Variable


def to_var(x):
    """
    The to_var function converts a Tensor to a Variable which supports
    backpropagation. It also moves the input to cuda if it is available.

    :param x: Convert the input to a tensor
    :return: A variable that contains the data in the tensor x
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
