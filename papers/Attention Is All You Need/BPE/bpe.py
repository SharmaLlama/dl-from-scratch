import pickle
from utils import _replace_encodings, _encode_utf8, _get_bigrams, _sum_bigrams

class BPEEncoder:
    def __init__(self, train_set = None, vocab=None):
        ## train set should be 2D list where each list is a sentence
        self.train_set = train_set
        self.vocab = vocab
        assert (self.vocab or self.train_set), "Both train set and vocab is None"
    
    def train(self, max_vocab_size = 100):
        if self.train_set:
            self.encoded_train_set = _encode_utf8(self.train_set)
            merges = {}
            curr_max = 256 if not self.vocab else max(self.vocab) + 1
            while curr_max - 255 <= max_vocab_size:
                stats = _sum_bigrams(self.encoded_train_set)
                if not stats:
                    break

                top_pair = max(stats, key=stats.get)
                if stats[top_pair] == 1: # we dont have any merges left
                    return merges
                
                self.encoded_train_set = list(map(lambda enc: _replace_encodings(enc, top_pair, 
                                curr_max), self.encoded_train_set))
                
                merges[(top_pair)] = curr_max
                curr_max += 1
                
            self.vocab = merges
        

    def save_vocab(self, fn):
        with open(fn , 'wb') as f:
            pickle.dump(self.vocab, f)

    
    def encode(self, text):
        if isinstance(text, str):
            text = [text]
            
        utf8_encoding = _encode_utf8(text)
        while True:
            stats = _sum_bigrams(utf8_encoding)
            pair = min(stats, key = lambda p: self.vocab.get(p, float("inf")))
            if pair not in self.vocab:
                break
            idx = self.vocab[pair]
            utf8_encoding = list(map(lambda enc: _replace_encodings(enc, pair, idx), 
                            utf8_encoding))
        return utf8_encoding


class BPEDecoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_reverse = {v:k for k,v in self.vocab.items()}

    def decode(self, text):
        if isinstance(text, str):
            text = [text]
        
        def __one_decode(sentence):
            original_text = []
            while True:
                non_seen = True
                for byte in sentence:
                    if byte in self.vocab_reverse:
                        original_text.append(self.vocab_reverse[byte][0])
                        original_text.append(self.vocab_reverse[byte][1])
                        non_seen = False
                    else: 
                        original_text += [byte]
                if non_seen:

                    return bytes(original_text).decode("utf-8", errors="replace")
                else: 
                    sentence = original_text
                    original_text = []

        return list(map(__one_decode, text))
