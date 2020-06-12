import nltk
from nltk import word_tokenize
import sys
nltk.download('punkt')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP

NP -> N | Det N | N PP | Det N PP | Det Adj N | Det Adj Adj Adj N

VP -> V | V NP | V PP | V NP PP | Adv V | Adv V NP | V Adv | V PP Adv

PP -> P NP | P S
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Separate words
    words = word_tokenize(sentence)
    
    # Remove non-alphabetic words
    for w in words: 
        alphabet_chars = 0
        
        for char in w:
            if char.isalpha(): 
                alphabet_chars += 1

        if alphabet_chars == 0: 
            words.remove(w)
                

    # Make all words lower case
    words = [x.lower() for x in words]

    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []

    # Find all noun phrases
    noun_phrases = []

    for child in tree.subtrees(lambda t: t.label() == 'NP'):     
        noun_phrases.append(child)

        # Check how many noun phrases are contained within each noun phrase (If 1 then no subtree is a noun phrase)
        if len(list(child.subtrees(lambda t: t.label() == 'NP'))) == 1: 
            np_chunks.append(child)

    return np_chunks

if __name__ == "__main__":
    main()
