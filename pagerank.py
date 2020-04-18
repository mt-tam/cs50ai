import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"\nPageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"\nPageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    # New dict to keep results
    page_results = {}

    # Add probability (random page) for each page in corpus
    for p in corpus:
        page_results[p] = (1 - damping_factor) / len(corpus)

    # Add probability (random link) to each page linked by the selected page
    for p in corpus:
        if p == page: 
            for l in corpus[page]:
                page_results[l] += damping_factor / len(corpus[page])

    return page_results
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # Keep track of frequency per page
    page_frequencies = {}
    for p in corpus: 
        page_frequencies[p] = 0
    
    # Pick first page randomly
    page = random.choice(list(corpus.keys()))
    page_frequencies[page] += 1

    # Repeat n-1 times (n being sample size)
    for i in range(n - 1): 
        
        # Pass page through transition model
        res = transition_model(corpus, page, DAMPING)
        
        # Get list of pages and probability distribution
        pages = list(res.keys())
        probabilities = list(res.values())

        # Pick next page based on probabilities
        page = random.choices(pages, probabilities)[0]
        page_frequencies[page] += 1

    # Diivide frequencies by Sample Size
    for i, j in page_frequencies.items(): 
        page_frequencies[i] = j / n

    return page_frequencies


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Non-damping factor probability
    first_prob = (1 - damping_factor) / len(corpus)

    # Keep track of frequency per page
    page_frequencies = {}

    # Set initial probabilities 
    for p in corpus: 
        page_frequencies[p] = 1 / len(corpus)
        if len(corpus[p]) == 0:
            corpus[p] = set(corpus.keys())

    # Repeat Once
    while True: 

        significant_mods = 0

        # Loop through each page
        for p in corpus: 

            # Find all pages linking to page
            linked_pages = []
            for i in corpus: 
                if p in corpus[i]:
                    linked_pages.append(i)

            # Sum ranks of pages linking to p
            temp_sum = 0
            for j in linked_pages:
                temp_sum += page_frequencies[j] / len(corpus[j])

            # Calculate new probability
            new_probability = first_prob + damping_factor * temp_sum

            # Keep track if significant modification was done
            if abs(new_probability - page_frequencies[p]) > 0.001:
                significant_mods += 1
                
            page_frequencies[p] = new_probability

        # If all values have not changed more than 0.001 than exit
        if significant_mods == 0: 
            break

    return page_frequencies


if __name__ == "__main__":
    main()
