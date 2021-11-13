import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
everyWords = data["everyWords"]
everyTags = data["everyTags"]
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)       # machine knows learned parameters
model.eval()

# Implement of actual chat
botName = "Bot"
print("Let's chat. Type 'quit' to exit.")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, everyWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = everyTags[predicted.item()]

    # Probability
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
                print()
    else:
        print(f"{botName}: I do not understand..")
        print()
