# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# %%
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

# def printLines(file, n=10):
#     with open(file, 'rb') as datafile:
#         lines = datafile.readlines()
#     for line in lines[:n]:
#         print(line)

# printLines(os.path.join(corpus, "movie_lines.txt"))


# %%
# # Splits each line of the file into a dictionary of fields
# def loadLines(fileName, fields):
#     lines = {}
#     with open(fileName, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             values = line.split(" +++$+++ ")
#             # Extract fields
#             lineObj = {}
#             for i, field in enumerate(fields):
#                 lineObj[field] = values[i]
#             lines[lineObj['lineID']] = lineObj
#     return lines


# # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
# def loadConversations(fileName, lines, fields):
#     conversations = []
#     with open(fileName, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             values = line.split(" +++$+++ ")
#             # Extract fields
#             convObj = {}
#             for i, field in enumerate(fields):
#                 convObj[field] = values[i]
#             # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
#             utterance_id_pattern = re.compile('L[0-9]+')
#             lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
#             # Reassemble lines
#             convObj["lines"] = []
#             for lineId in lineIds:
#                 convObj["lines"].append(lines[lineId])
#             conversations.append(convObj)
#     return conversations


# # Extracts pairs of sentences from conversations
# def extractSentencePairs(conversations):
#     qa_pairs = []
#     for conversation in conversations:
#         # Iterate over all the lines of the conversation
#         for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
#             inputLine = conversation["lines"][i]["text"].strip()
#             targetLine = conversation["lines"][i+1]["text"].strip()
#             # Filter wrong samples (if one of the lists is empty)
#             if inputLine and targetLine:
#                 qa_pairs.append([inputLine, targetLine])
#     return qa_pairs


# %%
# # Splits each line of the file into a dictionary of fields
# def loadLines(fileName, fields):
#     lines = {}
#     with open(fileName, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             values = line.split(" +++$+++ ")
#             # Extract fields
#             lineObj = {}
#             for i, field in enumerate(fields):
#                 lineObj[field] = values[i]
#             lines[lineObj['lineID']] = lineObj
#     return lines


# # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
# def loadConversations(fileName, lines, fields):
#     conversations = []
#     with open(fileName, 'r', encoding='iso-8859-1') as f:
#         for line in f:
#             values = line.split(" +++$+++ ")
#             # Extract fields
#             convObj = {}
#             for i, field in enumerate(fields):
#                 convObj[field] = values[i]
#             # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
#             utterance_id_pattern = re.compile('L[0-9]+')
#             lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
#             # Reassemble lines
#             convObj["lines"] = []
#             for lineId in lineIds:
#                 convObj["lines"].append(lines[lineId])
#             conversations.append(convObj)
#     return conversations


# # Extracts pairs of sentences from conversations
# def extractSentencePairs(conversations):
#     qa_pairs = []
#     for conversation in conversations:
#         # Iterate over all the lines of the conversation
#         for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
#             inputLine = conversation["lines"][i]["text"].strip()
#             targetLine = conversation["lines"][i+1]["text"].strip()
#             # Filter wrong samples (if one of the lists is empty)
#             if inputLine and targetLine:
#                 qa_pairs.append([inputLine, targetLine])
#     return qa_pairs


# %%
# # Define path to new file
# datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# delimiter = '\t'
# # Unescape the delimiter
# delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# # Initialize lines dict, conversations list, and field ids
# lines = {}
# conversations = []
# MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
# MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# # Load lines and process conversations
# print("\nProcessing corpus...")
# lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
# print("\nLoading conversations...")
# conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
#                                   lines, MOVIE_CONVERSATIONS_FIELDS)

# # Write new csv file
# print("\nWriting newly formatted file...")
# with open(datafile, 'w', encoding='utf-8') as outputfile:
#     writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
#     for pair in extractSentencePairs(conversations):
#         writer.writerow(pair)

# # Print a sample of lines
# print("\nSample lines from file:")
# printLines(datafile)


# %%
datafile = "splite_data/train_unclean_data"
test_datafile = "splite_data/test_data"
import json

MIN_LENGHT = 20
def load_data(path):
    lines = open(path).readlines()
    data = [json.loads(x) for x in lines]
    data2 = [item for item in data if len(item) > MIN_LENGHT]
    return data2

def split2line(l):
    split_token = [";", "{", "}"]
    result = []
    line = []
    for item in l:
        line.append(item)
        if item in split_token:
            result.append(line)
            line = []
    result.append(line)
    return result

def split2pair(l):
    result = []
    for i in range(1, len(l)):
        input = []
        for j in range(i):
            input += l[j]
        output = l[i]
        result.append((input.copy(),output.copy()))
    return result

def split2pair2(l):
    result = []
    inputstr = []
    for i in range(1, len(l)-1):
        inputstr += l[i - 1]
        outputstr = l[i]
        result.append((inputstr.copy(),outputstr.copy()))
    # for i in range(1, len(l)):
    #     input = []
    #     for j in range(i):
    #         input += l[j]
    #         for k in range(len(l[i])):
    #             input += l[i][:k]
    #             output = l[i][k:]
    #             result.append((input.copy(),output.copy()))
    return result
    
def wrap_s2p(raw_data):
    data5 = []
    for i, item in enumerate(raw_data):
        data5 += split2pair(item)
    return [(" ".join(line[0]), " ".join(line[1])) for line in data5]

def wrap_s2p2(raw_data):
    data5 = []
    for i, item in enumerate(raw_data):
        data5 += split2pair2(item)
        # print(len(data5))
        if len(data5) > 100000:
            break
    return [(" ".join(line[0]), " ".join(line[1])) for line in data5]


# %%
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# %%
MAX_LENGTH = 20  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def readVocs(datafile,test_datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    train_data = load_data(datafile)
    train_data = [split2line(item) for item in train_data]
    train_data = wrap_s2p2(train_data)
    # lines = open(datafile, encoding='utf-8').\
    #     read().strip().split('\n')
    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = train_data
    test_data = load_data(test_datafile)
    test_data = [split2line(item) for item in test_data]
    test_data = wrap_s2p2(test_data)
    test_pairs = test_data
    voc = Voc(corpus_name)
    return voc, pairs, test_pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile,test_datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs, test_pairs = readVocs(datafile,test_datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    test_pairs = filterPairs(test_pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs + test_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs, test_pairs


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs, test_pairs = loadPrepareData(corpus, corpus_name, datafile,test_datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# %%
"getQualifiers" in voc.word2index.keys()


# %%
MIN_COUNT = 30    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)
test_pairs = trimRareWords(voc, test_pairs, MIN_COUNT)


# %%
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# %%
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# %%
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# %%
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# %%
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
    


# %%
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# %%
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("\nIteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0
            # print("start test")
            encoder.eval()
            decoder.eval()
            print("Testing...")
            searcher = GreedySearchDecoder(encoder, decoder)
            train_bleu_score, train_beam_score = evaluateTest(pairs, encoder, decoder, searcher, voc)
            
            test_bleu_score, test_beam_score = evaluateTest(test_pairs, encoder, decoder, searcher, voc)
            print("train bleu score:{}".format(train_bleu_score))
            print("train beam score:{}".format(train_beam_score))
            print("test bleu score:{}".format(test_bleu_score))
            print("test beam score:{}".format(test_beam_score))
            print(" ")
            encoder.train()
            decoder.train()


        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# %%
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# %%
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

def evaluateSentence(input_sentence, encoder, decoder, searcher, voc):
    input_sentence = normalizeString(input_sentence)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    # Format and print response sentence
    new_output_words = []
    for x in output_words:
        if x == "EOS" or x == "PAD":
            break
        new_output_words.append(x)
    return " ".join(new_output_words)


def get_bleu_result(candidate, references):
    candidate = " ".join([candidate, candidate, candidate, candidate])

    references = [" ".join([reference, reference, reference, reference]) for reference in references]
    references = [reference .split(" ") for reference in references]
    candidate = [item for item in candidate.split(" ")]
    score = sentence_bleu(references, candidate)
    # print(reference, candidate)
    return score

def evaluateTest(pairs, encoder, decoder, searcher, voc):
    scores = []
    beam_scores = []
    new_pair = [random.choice(pairs) for _ in range(200)]
    
 
    for i in tqdm(range(len(new_pair))):
        input_sentence, output_sentence = new_pair[i]
        
  
        try:
            beam_output_words = beam_evaluate(encoder, decoder, searcher, voc, input_sentence)
            bss = [get_bleu_result(beam2tokens(end_word, voc), [output_sentence]) for end_word in beam_output_words]
            # print(bss)
            beam_score = np.max(bss)
            # beam_output_words = [beam2tokens(end_word, voc) for end_word in beam_output_words]
            output_words = evaluateSentence(input_sentence, encoder, decoder, searcher, voc)
            # print("IN:\t{}".format(input_sentence))
            # print("PRED:\t{}".format(output_words))
            # print("REAL:\t{}".format(output_sentence))
            bleu_score = get_bleu_result(output_words, [output_sentence])
            scores.append(bleu_score)
            beam_scores.append(beam_score)
        except:
            scores.append(0)
            beam_scores.append(0)
            pass
    return np.mean(scores), np.mean(beam_scores)
    # return scores



# %%
import queue

class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.log_prob = log_prob
        self.length = length  

def beam_evaluate(encoder, decoder, searcher, voc, sentence, beam_width = 5, max_length = MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    input_length = lengths.to(device)### Format input sentence as a batch
    

    encoder_outputs, encoder_hidden = encoder(input_batch, input_length)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_attentions = None
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    root = Node(decoder_hidden, None, decoder_input, decoder_attentions, 0, 1)
    q = queue.Queue()
    q.put(root)
    
    end_nodes = [] #最终节点的位置，用于回溯

    while not q.empty():
            candidates = []  #每一层的可能被拓展的节点，只需选取每个父节点的儿子节点中概率最大的k个即可
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                decoder_hidden = node.hidden
                decoder_attentions = node.attn
                length = node.length
                # 搜索终止条件
                if decoder_input.item() == EOS_token or decoder_input.item() == PAD_token or node.length >= max_length:
                    end_nodes.append(node)
                    continue
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                decoder_input = torch.unsqueeze(decoder_input, 0) 
                log_prob, indices = decoder_output.data.topk(beam_width) #选取某个父节点的儿子节点概率最大的k个
                # print(log_prob, indices)
                # print(log_prob[0], indices[0])
                for k in range(beam_width):
                      index = indices[0][k].unsqueeze(0).unsqueeze(0)
                    #   print(indices)
                      log_p = log_prob[0][k].item()
                      child = Node(decoder_hidden, node, index, decoder_attentions, node.log_prob + log_p, length + 1)
                      candidates.append((node.log_prob + log_p, child))  #建立候选儿子节点，注意这里概率需要累计
                      # print((node.log_prob + log_p, child)
            candidates = sorted(candidates, key=lambda x:x[0]/x[1].length, reverse=True) #候选节点排序
            length = min(len(candidates), beam_width)  #取前k个，如果不足k个，则全部入选
            # print(candidates)
            # print("\\")
            for i in range(length):              
                q.put(candidates[i][1])
    end_nodes = sorted(end_nodes, key=lambda x:x.log_prob/x.length, reverse=True) #候选节点排序
    # for node in end_nodes:
    #     print(node.log_prob)
    return end_nodes[:beam_width]

def beam2tokens(end_node, voc):
    words = []
    node = end_node
    while node.previous_node != None:
        index = node.decoder_input
        word = voc.index2word[index.item()]
        words.insert(0, word)
        node = node.previous_node
    return " ".join(words)



def beam_predict(encoder, decoder, beam, pair, lang, show = True):
    bs = beam_search(encoder, decoder, pair[0], lang, beam)
    ss = [beam2tokens(node, lang) for node in bs]
    vs = [node.log_prob for node in bs]
    scores = [get_bleu_result(" ".join(s), pair[1]) for s in ss]
    max_score = max(scores)
    if show:
        max_index = scores.index(max(scores))
        print('>', pair[0])
        print('=', pair[1])
        print('<', " ".join(ss[max_index][:-1]), vs[max_index])
        # for (index, s) in enumerate(ss):
        #     print('<', " ".join(s[:-1]), vs[index])
        print("validation beam bleu score",max_score)
    return max_score, vs


# %%
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 32

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# %%
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 2000
print_every = 5
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)


# %%
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)


# %%
evaluateInput(encoder, decoder, searcher, voc)


# %%
evaluateTest(test_pairs, encoder, decoder, searcher, voc)


# %%



# %%


