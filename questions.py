import nltk
import sys
import os
import string
import math
#nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 3


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    
    new_dict = dict()

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            new_dict[filename] = contents

    return new_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Separate and lowercase words
    words = [word.lower() for word in nltk.word_tokenize(document)]

    # Filter stopwords
    words = [word for word in words if not word in nltk.corpus.stopwords.words("english")]

    # Filter punctuation
    words = [word for word in words if not word in string.punctuation]
    return words
    

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Calculate IDFs
    idfs = dict()

    for filename, words in documents.items():
        for word in words:
            # document frequency
            f = sum(word in documents[filename] for filename in documents)
            
            # inverse doc frequency
            idf = math.log(len(documents) / f)
            idfs[word] = idf

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    new_dict = dict()
    
    # For each file, calculate the sum of all words' tf-idf score
    for file in files: 
        new_dict[file] = 0
        
        for word in query: 

            if word in files[file]: 

                # calculate term frequency (tf)
                tf = files[file].count(word) 

                # retrieve inverse document frequency (idf)
                idf = idfs[word]

                # compute tf-idf 
                tf_idf = tf * idf

                # add tf-idf to new dict's total file score
                new_dict[file] += tf_idf              

    ranked_files = sorted(new_dict, key=new_dict.get, reverse = True)
    #print(ranked_files)
    return ranked_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    new_dict = dict()
    
    # For each sentence, calculate the sum of all words' idf score
    for sentence in sentences: 
        new_dict[sentence] = 0
        
        for word in query: 

            if word in sentences[sentence]: 

                # retrieve inverse document frequency (idf)
                idf = idfs[word]

                # add idf to new dict's total sentence score
                new_dict[sentence] += idf            

    ranked_sentences = sorted(new_dict, key=new_dict.get, reverse = True)
    #print(ranked_sentences[:n])

    # recreate a dict with top idf sentences
    temp_dict = dict()
    for sentence in ranked_sentences[:n]: 
        idf = new_dict[sentence]
        temp_dict[sentence] = idf

    # check if idf score is a tie
    if len(temp_dict.values()) == len(set(temp_dict.values())): 
        return ranked_sentences[:n] # return initial rank if no duplicates  

    # else, calculate query term density (QTD)
    else:
        qtd_dict = dict() 
        for sentence in temp_dict: 
            intersection = [word for word in sentences[sentence] if word in query] # find all matching words
            query_density = len(intersection) / len(sentences[sentence]) # calculate density
            qtd_dict[sentence] = query_density

    ranked_sentences = sorted(qtd_dict, key=qtd_dict.get, reverse = True)
    #print(ranked_sentences[:n])

    return ranked_sentences[:n]


if __name__ == "__main__":
    main()
