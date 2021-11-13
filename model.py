import torch
import torch.nn as nn

#  NeuralNet class is derived from nn.Module
# Our model is a feedforward model with 3 layers. So total 4 layers: input, hidden, hidden
class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        """
        inputSize = length of bag of words(must be fixed)
        hiddenSize = (you can set it as you wish)
        numclasses = number of different tags (must be fixed)
        """
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        # self.l3 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, numClasses)

        # activation function. works in between the layers
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        The implimantation of feeding forward layers.(As it is a feed forward model)
        """

        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # out = self.l4(out)
        # out = self.relu(out)

        # return the output for applying softmax(probability)
        return out
