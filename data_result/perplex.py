import math
import torch
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

strings = [
           "<|endoftext|>My dog died after getting stuck in a tree. I have had the same bug with a man in power and that power power had power on power go into power go power down power down power down power power down power power down. This has caused power loss power down",
           "<|endoftext|>The potato is probably the world's most widely eaten plant. But what if it's also the most dangerous? In the last two decades, there's been a dramatic decrease in potato crop damage from crop rot and disease. The decline, which started in",
           
]

input = read_dialog('./../output.txt')

print([score(s) for s in input])