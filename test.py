from operator import add
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.file_utils import cached_path

BIG_CONST = 1e10

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices

def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        # only keep words in the Bag of Words that tokenize to 1 token
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)

device = 'cpu'
pretrained_model = 'gpt2-medium'

model = GPT2LMHeadModel.from_pretrained(
    pretrained_model,
    output_hidden_states=True
)
model.to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

raw_text = "The man"

context = tokenizer.encode(
    tokenizer.bos_token + raw_text,
    add_special_tokens=False
)

print(context)


# gpt2-medium has 24 heads (multi-headed transformers), so
# past and all_hidden is a tuple of size 24
def run_normal():
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    for i in range(10):
        logits, past, all_hidden = model(output_so_far)

        temperature = 1.0
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        last = torch.multinomial(probs, num_samples=1)

        output_so_far = torch.cat((output_so_far, last), dim=1)

    print(tokenizer.decode(output_so_far.tolist()[0]))

# run_normal()

# here we are essentially backpropagating from the prediction value to the
# "past" key-value pairs that act as history for the transformer

def run_bow():
    bag_of_words = 'military'
    bow_indices = get_bag_of_words_indices(bag_of_words.split(";"), tokenizer)
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device=device)

    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t
    
    past = None
    last = None
    for _ in range(10):
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])
        # logits, past, all_hidden = model(output_so_far)

        # temperature = 1.0
        # logits = logits[:, -1, :] / temperature
        # probs = F.softmax(logits, dim=-1)

        # last = torch.multinomial(probs, num_samples=1)

        # output_so_far = torch.cat((output_so_far, last), dim=1)

        grad_accumulator = [
            (np.zeros(p.shape).astype("float32"))
            for p in past
        ]

        # there is no window
        window_mask = torch.ones_like(past[0]).to(device)

        for i in range(3):
            print(i)
            curr_perturbation = [
                to_var(torch.from_numpy(p_), requires_grad=True, device=device)
                for p_ in grad_accumulator
            ]

            perturbed_past = list(map(add, past, curr_perturbation))
            all_logits, _, all_hidden = model(last, past=perturbed_past)
            logits = all_logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            loss = 0.0

            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss

            loss.backward()

            grad_norms = [
                (torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]

            stepsize = 0.03
            gamma = 1.5

            # normalize gradients
            grad = [
                -stepsize *
                (p_.grad * window_mask / grad_norms[
                    index] ** gamma).data.cpu().numpy()
                for index, p_ in enumerate(curr_perturbation)
            ]

            # accumulate gradient
            grad_accumulator = list(map(add, grad, grad_accumulator))

            # reset gradients, just to make sure
            for p_ in curr_perturbation:
                p_.grad.data.zero_()

            # removing past from the graph
            new_past = []
            for p_ in past:
                new_past.append(p_.detach())
            past = new_past

        grad_accumulator = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]
        pert_past = list(map(add, past, grad_accumulator))

        all_logits, past, _ = model(last, past=pert_past)

        last = torch.multinomial(probs, num_samples=1)
        output_so_far = torch.cat((output_so_far, last), dim=1)

    print(tokenizer.decode(output_so_far.tolist()[0]))

run_bow()

# print(logits.shape)
# print(past[0].shape)
# print(all_hidden[0].shape)

# logits = top_k_filter(logits, k=10)
# pert_probs = F.softmax(logits, dim=-1)

# pert_logits, past, pert_all_hidden = model(last, past=pert_past)
# pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
# pert_probs = F.softmax(pert_logits, dim=-1)

# last = output_so_far[:, -1:]

# _, past, _ = model(output_so_far[:, :-1])
# pert_past = past


