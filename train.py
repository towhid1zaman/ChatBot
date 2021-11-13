import json
import numpy as np
from torch.nn.modules import loss
from nltk_utils import tokenize, stem, bag_of_words

# for creating chatbot dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# for model
from model import NeuralNet

# 
# The main train section start from here
# 
with open('intents.json', 'r') as f:
    intents = json.load(f)

everyWords = []
everyTags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    everyTags.append(tag)

    # Get words
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        everyWords.extend(w)
        # add to xy pair
        xy.append((w, tag))     # xy list keeps petterns with corresponing tag
    

# stemming and exclude punctuation characters
ignoreChars = ['?', '.', ',', '.', '!', ':', '=', '+', '-']
everyWords = [stem(word) for word in everyWords if word not in ignoreChars]

# sort the words and remove duplicate words
everyWords = sorted(set(everyWords))

# sort and remove duplicate from Tags
everyTags = sorted(set(everyTags))

xTrain = []
yTrain = []
for (patternSentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(patternSentence, everyWords)
    xTrain.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = everyTags.index(tag)
    yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

# Hyperperameters
batchSize = 8
hiddenSize = 8
inputSize = len(xTrain[0])
outputSize = len(everyTags)
learningRate = 0.001
numEpochs = 1000

# Implement Dataset
class ChatDataSet(Dataset):
    def __init__(self):
        self.numberOfSamples = len(xTrain)
        self.xData = xTrain
        self.yData = yTrain
    
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.xData[index], self.yData[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.numberOfSamples


dataset = ChatDataSet()
trainLoader = DataLoader(dataset = dataset, batch_size=batchSize, shuffle=True, num_workers=0)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

# Actual training loop
for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        # push word to device
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass optimizer
        # before backward we have to empty the gradient first
        optimizer.zero_grad()
        loss.backward()         # to calculate backpropagation
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{numEpochs}, Loss = {loss.item():.4f}")

print(f"Final Loss = {loss.item():.4f}")

# Save the data
data = {
    "modelState": model.state_dict(),
    "inputSize" : inputSize,
    "outputSize": outputSize,
    "hiddenSize": hiddenSize,
    "everyWords": everyWords,
    "everyTags": everyTags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Trainig Complete. File Saved To {FILE}")