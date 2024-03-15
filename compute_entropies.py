import pm4py
import pandas as pd
import math
from collections import Counter
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100000)
pd.set_option('display.max_rows', 500)

CASE_ID = "case:concept:name"
EVENT_ID = "concept:name"
TIMESTAMP = "time:timestamp"



def entropy(prob_dict):
    return sum([-v*math.log2(v) for _, v in prob_dict.items()])

def compute_entropy_from_ngrams(ngrams):
    freq = Counter(ngrams)
    total = len(ngrams)
    prob = {word: freq/total for word, freq in freq.items()}
    return entropy(prob)


def build_ngrams_from_text(text, n):
    ngrams = []

    i = 0
    while i + n <= len(text):
        ngrams.append(' '.join(text[i:i + n]))
        i += 1

    return ngrams

def compute_ngram_entropy(text, n):
    ngrams = build_ngrams_from_text(text, n)
    ngram_entropy = compute_entropy_from_ngrams(ngrams)

    print(f'{n}-gram entropy: = ', ngram_entropy)

    return ngram_entropy

def compute_ngram_entropy_rate(text, n):
    ngram_entropy = compute_ngram_entropy(text, n)
    entropy_rate = ngram_entropy / n

    return entropy_rate

def compute_entropy_rate_series(text, max_N=5):
    ngram_entropies = {}
    for n in range(1, max_N + 1):
        entropy_rate = compute_ngram_entropy_rate(text, n)
        ngram_entropies[n] = entropy_rate

        print(f'{n}-gram entropy rate: = ', entropy_rate)

    return ngram_entropies



"""Event Log Entropies: Back, C.O., Debois, S. & Slaats, T. Entropy as a Measure of Log Variability. J Data Semant 8, 129–156 (2019). https://doi.org/10.1007/s13740-019-00105-3"""


def compute_k_block_entropy(log, k:int):
    """
    Definition 10 and 12

    :param log:
    :param k:
    :param flatten:
    :return:
    """

    block_dict = {}
    cases = log.groupby(by=[CASE_ID])
    for cid, case in cases:
        #Use blocks of length [0, ..., len(case)-k]
        act_sequence = case[EVENT_ID].tolist()
        for i in range(len(case)-k+1):
            k_block = "***".join(act_sequence[i:i+k])

            if k_block in block_dict.keys():
                block_dict[k_block] += 1
            else:
                block_dict[k_block] = 1

    block_count = sum(block_dict.values())
    probabilities = {b: (c / block_count) for b, c in block_dict.items()}

    return entropy(probabilities)

def compute_k_block_entropy_rate(log, k:int=200):
    """
    Definition 21
    """
    rate = compute_k_block_entropy(log, k=k) / k
    return rate


def compute_block_entropy_rate_series(log, max_n=12):
    block_entropy_rates = {}

    for k in range(1, max_n + 1):
        block_entropy_rate = compute_k_block_entropy_rate(log, k=k)
        print(f'{k}-block entropy rate: = ', block_entropy_rate)

        block_entropy_rates[k] = block_entropy_rate

    return block_entropy_rates


def strict_constraint(n, h, N, alph_size):
    """4.10"""
    left = N * h
    right = n * math.pow(2, (n * h)) * math.log2(alph_size)
    return left > right


log = pm4py.read_xes("../bpi_challenge_incidents.xes") #Load the log you want
print(log.head(100))

trace_seq = log[EVENT_ID].tolist()
event_seq = log.sort_values(by=[TIMESTAMP])[EVENT_ID].tolist()
vocab_size = log[EVENT_ID].nunique()
max_case_length = max([len(c) for id, c in log.groupby(by=[CASE_ID])])

max_N = 20

h_rate_trace = compute_entropy_rate_series(text=trace_seq, max_N=max_N)
h_rate_events = compute_entropy_rate_series(text=event_seq, max_N=max_N)
h_rate_el = compute_block_entropy_rate_series(log, max_n=max_N)

thres_trace, thres_event, thres_el = 0, 0, 0

for n in range(1, max_N):
    if not strict_constraint(n, h_rate_trace[n], len(trace_seq), vocab_size):
        thres_trace = n - 1
        break

for n in range(1, max_N):
    if not strict_constraint(n, h_rate_events[n], len(event_seq), vocab_size):
        thres_event = n - 1
        break

for n in range(1, max_N):
    if not strict_constraint(n, h_rate_el[n], max_case_length, vocab_size):
        thres_el = n - 1
        break

print(thres_trace, thres_event, thres_el)
