from functools import reduce
from collections import Counter


def _encode_utf8(sentences):
    """
    Given a list of sentences, this function encodes them into utf-8. 
    Expects a 2D list input.
    """ 
    utf = [sentence.encode("utf-8") for sentence in sentences]
    tokens = [list(map(int, u)) for u in utf]
    return tokens

def _get_bigrams(sentence):
    """
    Gets the bi-gram for a given sentence.
    """
    return dict(Counter(zip(sentence[:-1], sentence[1:])))

def _sum_dicts(dicts):
    """
    Calculates the sum of n dictionaries. If the keys match, then the entries are summed,
    else we just set it to whatever the value is.
    """
    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator
    return reduce(reducer, dicts, {})

def _sum_bigrams(sentences):
    """
    Sums the bi-grams for all sentences.
    """
    pair_counts = [_get_bigrams(sentence) for sentence in sentences]
    return _sum_dicts(pair_counts)


def _replace_encodings(encoding, pair, new_idx):
    """
    Given the current encoding of a sentence, replace the bi-gram with the new index provided.
    """
    new_encoding = []
    i = 0
    while i < len(encoding):
        if i < len(encoding) - 1 and encoding[i] == pair[0] and encoding[i + 1] == pair[1]:
            new_encoding.append(new_idx)
            i += 2
        else: 
            new_encoding.append(encoding[i])
            i += 1
    return new_encoding