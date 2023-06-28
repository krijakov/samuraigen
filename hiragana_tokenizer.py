#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Description: tokenizer for romanized Japanese text using the romanized hiragana alphabet characters as tokens.

'''

from pykakasi import kakasi


class HiraganaTokenizer:
    """Tokenizer for romanized Japanese text using the romanized hiragana alphabet characters as tokens """
    def __init__(self):
        self.hiraganas = [
        "あ", "い", "う", "え", "お",
        "か", "き", "く", "け", "こ",
        "さ", "し", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と",
        "な", "に", "ぬ", "ね", "の",
        "は", "ひ", "ふ", "へ", "ほ",
        "ま", "み", "む", "め", "も",
        "や", "ゆ", "よ",
        "ら", "り", "る", "れ", "ろ",
        "わ", "を", "ん",
        "が", "ぎ", "ぐ", "げ", "ご",
        "ざ", "じ", "ず", "ぜ", "ぞ",
        "だ", "ぢ", "づ", "で", "ど",
        "ば", "び", "ぶ", "べ", "ぼ",
        "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
        "きゃ", "きゅ", "きょ",
        "しゃ", "しゅ", "しょ",
        "ちゃ", "ちゅ", "ちょ",
        "にゃ", "にゅ", "にょ",
        "ひゃ", "ひゅ", "ひょ",
        "みゃ", "みゅ", "みょ",
        "りゃ", "りゅ", "りょ",
        "ぎゃ", "ぎゅ", "ぎょ",
        "じゃ", "じゅ", "じょ",
        "びゃ", "びゅ", "びょ",
        "ぴゃ", "ぴゅ", "ぴょ",
        "ん",
        ] # list of hiraganas, by chatgpt :O
        self.tokens = []
        # note: multiple hiraganas seem to have the same romanization, but we need every token only once:
        for h in self.hiraganas:
            c = kakasi().convert(h)[0]['hepburn']
            if c not in self.tokens:
                self.tokens.append(c)

        # add special characters:
        self.tokens.insert(0, " ")
        self.tokens.insert(0, "\n")

        self.max_token_length = max(len(t) for t in self.tokens)
        self.vocab_size = len(self.tokens)
        # if the token set has been reduced:
        self.IsReduced = False

        # lookup tables:
        self.stoi = {s:i for i,s in enumerate(self.tokens)}
        self.itos = {i:s for s, i in self.stoi.items()}

    def __call__(self, word:str) -> list:
        """Encodes a string, returning a list of integers according to the lookup table."""
        word_tokenized = []
        window = ""
        for ch in word:
            window += ch
            if window in self.tokens:
                word_tokenized.append(self.stoi[window])
                window = ""
            if len(window) > self.max_token_length:
                #print(f"Failed to tokenize: {word}, Unknown token: {window}")
                window = ""
                return []
        return word_tokenized
       
        
    def decode(self, toks:list, joined=True) -> str|list:
        """Decodes a list of integers into a string if joined == True, a list of characters otherwise."""
        word = []
        for t in toks:
            word.append(self.itos[t]) 
        return ''.join(word) if joined else word

    def reduceTokens(self, sample:list):
        """Reduces the vocabulary size by removing the unused characters from the tokens and lookup tables. 
            Input: list of token integers"""
        # get the occurences of various tokens:
        freqs = [0] * self.vocab_size
        for i in sample:
            for j, t in enumerate(self.tokens):
                if i == self.stoi[t]:
                    freqs[j] += 1 

        # remove the tokens from the vocabulary which don't occur in the sample:
        zero_places = []
        for i, f in enumerate(freqs):
            if f == 0:
                zero_places.append(i)
        new_tokens = [self.tokens[i] for i in range(self.vocab_size) if i not in sorted(zero_places)]
        self.tokens = new_tokens

        # update the lookup tables and variables:
        self.max_token_length = max(len(t) for t in self.tokens)
        self.vocab_size = len(self.tokens)
        # if the token set has been reduced:
        self.IsReduced = True
        # lookup tables:
        self.stoi = {s:i for i,s in enumerate(self.tokens)}
        self.itos = {i:s for s, i in self.stoi.items()}

        print(f"Removed {len(zero_places)} unused tokens, the new vocabulary size: {self.vocab_size}")

        
