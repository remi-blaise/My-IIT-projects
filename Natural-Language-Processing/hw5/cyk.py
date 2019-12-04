"""Read grammar definition and sentence to parse,
then output a valid parse for the sentence, given the grammar.

Implementation of Cocke-Younger-Kasami parsing
"""

import argparse
from collections import namedtuple
import re

# Data structures for rules
#  Nonterminal rules have one symbol on left-hand side, two symbols on right-hand side
NonterminalRule = namedtuple("NonterminalRule", ["lhs", "rhs1", "rhs2"])
#  Nonterminal rules have one symbol on left-hand side, one symbol on right-hand side
TerminalRule = namedtuple("TerminalRule", ["lhs", "rhs"])

# Data structure for parsed phrase
ParsedPhrase = namedtuple("ParsedPhrase", ["label", "children"])

def parse_rules(infile):
    """Parse grammar file with phrase structure rules, and
    return a tuple (nt, t), where nt is a list of nonterminal
    rules and t is a list of terminal rules
    """
    nt = []
    t = []
    ntmatcher = re.compile(r"^\s*(\w+)\s+->\s+(\w+)\s+(\w+)\s*$")
    tmatcher = re.compile(r"^\s*(\w+)\s+->\s+(\w+)\s*$")
    with open(infile) as input_text:
        for line in input_text:
            found = ntmatcher.search(line)
            if found:
                nt.append(NonterminalRule(*found.group(1, 2, 3)))
            else:
                found = tmatcher.search(line)
                if found:
                    t.append(TerminalRule(*found.group(1, 2)))
    return nt, t


def read_sentences(infile):
    """Read a file with one sentence per line, and return
    a list of word lists (one for each sentence)
    """
    with open(infile) as input_text:
        return [line.strip().split() for line in input_text if line]


def parse_sentence(nt_rules, t_rules, words):
    """Parse a sentence with the CYK algorithm

    :param nt_rules: List of nonterminal rules in grammar
    :param t_rules: List of terminal rules in grammar
    :param words: sequence (list) of words in sentence to parse
    :return: Recursively-nested NonterminalRule object representing parse tree
      (or None if parsing fails)
    """
    # NOTE -- you can change this data structure / function if you prefer to do
    # this differently, but the function still needs to return
    #   - a parse represented as recursively nested NonterminalRule / TerminalRule objects
    #   - or None if the sentence cannot be parsed

    # chart[m][n][symb] will contain a ParsedPhrase object for a phrase
    #  - of length m+1
    #  - starting at word n
    #  - of phrase category symb
    chart = [[{} for j in range(len(words))] for i in range(len(words))]

    # Initialize nonterminals in chart
    for i, word in enumerate(words):
        for tr in t_rules:
            if word == tr.rhs:
                chart[0][i][tr.lhs] = ParsedPhrase(label=tr.lhs, children=[word])

    # Work up the chart
    N = len(words)
    for m in range(1, N):
        for n in range(0, N - m):
            for k in range(n + 1, n + m + 1):
                for nt in nt_rules:
                    if nt.rhs1 in chart[k - n - 1][n] and nt.rhs2 in chart[n + m - k][k]:
                        chart[m][n][nt.lhs] = ParsedPhrase(label=nt.lhs, children=[ chart[k - n - 1][n][nt.rhs1], chart[n + m - k][k][nt.rhs2] ])

    # All valid sentence parses should have the label "S"
    return chart[len(words)-1][0].get("S")


def parse_to_string(parse):
    """Return a string representation of a parsed phrase object
    """
    if len(parse.children) > 1:
        return f"({parse.label} {parse_to_string(parse.children[0])} {parse_to_string(parse.children[1])})"
    return f"({parse.label} {parse.children[0]})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_file", default="grammar_1.txt")
    parser.add_argument("--sentence_file", default="sentences_1.txt")
    args = parser.parse_args()

    nt_rules, t_rules = parse_rules(args.grammar_file)
    word_sequences = read_sentences(args.sentence_file)

    for s in word_sequences:
        parse = parse_sentence(nt_rules, t_rules, s)
        if parse:
            print(parse_to_string(parse))
        else:
            print("Parsing failed")
