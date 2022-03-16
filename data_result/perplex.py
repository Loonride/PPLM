import math
import torch
import statistics
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.eval()

# could run with [score(s) for s in strings]

def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with open(file) as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return contents

def score(sent):
    indexed_tokens = tokenizer.encode(sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model.forward(tokens_tensor, labels=tokens_tensor)
    loss = outputs[0]
    return math.exp(loss.item())

pos_input = read_dialog('./../data/positive.txt')
neg_input = read_dialog('./../data/negative.txt')
neg2pos_input = read_dialog('./../data/negative_to_positive.txt')
pos2neg_input = read_dialog('./../data/positive_to_negative.txt')

two_input = []
four_input = []

for inp in pos_input:
    two_input.append(inp)
    four_input.append(inp)

for inp in neg_input:
    two_input.append(inp)
    four_input.append(inp)

for inp in neg2pos_input:
    four_input.append(inp)

for inp in pos2neg_input:
    four_input.append(inp)


pos_scores = [score(s) for s in pos_input]
neg_scores = [score(s) for s in neg_input]
pos2neg_scores = [score(s) for s in pos2neg_input]
neg2pos_scores = [score(s) for s in neg2pos_input]
two_input_scores = [score(s) for s in two_input]
four_input_scores = [score(s) for s in four_input]
print(statistics.mean(pos_scores))
print(statistics.stdev(pos_scores))
print(statistics.mean(neg_scores))
print(statistics.stdev(neg_scores))
print(statistics.mean(pos2neg_scores))
print(statistics.stdev(pos2neg_scores))
print(statistics.mean(neg2pos_scores))
print(statistics.stdev(neg2pos_scores))
print(statistics.mean(two_input_scores))
print(statistics.stdev(two_input_scores))
print(statistics.mean(four_input_scores))
print(statistics.stdev(four_input_scores))