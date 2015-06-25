"""Utilities for defining language models."""

import collections

def read_data(infile):
    """Read data from infile and convert it to lists of tokens ending with </s>."""
    if isinstance(infile, str):
        infile = open(infile)
    data = []
    for line in infile:
        words = line.split()
        data.append(words + ['</s>'])
    return data

def ngrams(data, n):
    """Convert data (as returned by read_data) into a list of n-grams."""
    result = []
    for words in data:
        words = (n-1)*['<s>'] + words
        for i in xrange(len(words)-n+1):
            result.append(words[i:i+n])
    return result

class Numberizer(object):
    def __init__(self, words, unk="<unk>"):
        """Make a numberizer.
        
        words: A vocabulary as returned by make_vocab."""

        self.w = list(sorted(set(words) | {unk}))
        self.n = {word:number for number, word in enumerate(self.w)}
        self.unk = self.n[unk]

    def numberize(self, word):
        """Convert a word into a number."""
        return self.n.get(word, self.unk)

def make_vocab(data, size=None, special=['<s>', '</s>', '<unk>']):
    """Make a vocabulary.

    data: Data as returned by read_data
    size: Limit vocabulary to this many types (optional)
    """
    c = collections.Counter()
    for words in data:
        for word in words:
            c[word] += 1

    for word in special:
        if word in c:
            del c[word]

    if size:
        vocab = {word for word, count in c.most_common(size-len(special))}
    else:
        vocab = set(c)
    vocab.update(special)

    return vocab

