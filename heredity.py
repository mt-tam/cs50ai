import csv
import itertools
import sys
import math
from functools import reduce

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue
    
        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                #print(f"> One Gene: {one_gene} - Two Genes: {two_genes} - Has Trait: {have_trait}")
                p = joint_probability(people, one_gene, two_genes, have_trait)
                #print(f"Join Probability: {p}")
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = []

    for p in people:
        p = people[p]
        name = p['name']
        mother = p['mother']
        father = p['father']
        
        # Figure out number of genes & whether trait or not
        no_genes = 1 if name in one_gene else 2 if name in two_genes else 0
        trait = True if name in have_trait else False

        # If no parents, then use unconditional probabilities
        if not mother or not father:
            gene_prob = PROBS['gene'][no_genes]
            trait_prob = PROBS['trait'][no_genes][trait]

        # Else, if parents, calculate the likelihood of getting the gene
        else:
            # Calculate the chance of receiving the gene from the mother
            mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
            mother_trait = True if mother in have_trait else False
            prob_mother = 0.5 - PROBS['mutation'] if mother_genes == 1 else 1 - PROBS['mutation'] if mother_genes == 2 else PROBS['mutation']

            # Calculate the chance of receiving the gene from the father
            father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
            father_trait = True if father in have_trait else False
            prob_father = 0.5 - PROBS['mutation'] if father_genes == 1 else 1 - PROBS['mutation'] if father_genes == 2 else PROBS['mutation']            

            # Calculate the probability that child has 0, 1 or 2 genes
            child_one_gene = prob_mother * (1 - prob_father) + (1 - prob_mother) * prob_father
            child_two_genes = prob_mother * prob_father
            child_zero_genes = 1 - child_one_gene - child_two_genes

            gene_prob = child_one_gene if no_genes == 1 else child_two_genes if no_genes == 2 else child_zero_genes
            trait_prob = PROBS['trait'][no_genes][trait]

        # Calculate total probability for person having selected # of genes + trait
        total_prob = gene_prob * trait_prob
        probabilities.append(total_prob)  

    # Calculate FINAL joint probability (by multiplying all values)
    joint_probability = reduce((lambda x, y: x * y), probabilities) 
    
    return joint_probability

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:

        # Update probability of having # of genes
        if name in one_gene:
            probabilities[name]["gene"][1] = p
        elif name in two_genes:
            probabilities[name]["gene"][2] = p
        else:
            probabilities[name]["gene"][0] = p

        # Update probability of having trait
        if name in have_trait:
            probabilities[name]["trait"][True] = p
        else:
            probabilities[name]["trait"][False] = p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities: 
        total_gene = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        total_trait = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]

        probabilities[person]["gene"][0] = (probabilities[person]["gene"][0] / total_gene) * 1
        probabilities[person]["gene"][1] = (probabilities[person]["gene"][1] / total_gene) * 1
        probabilities[person]["gene"][2] = (probabilities[person]["gene"][2] / total_gene) * 1

        probabilities[person]["trait"][True] = (probabilities[person]["trait"][True] / total_trait) * 1
        probabilities[person]["trait"][False] = (probabilities[person]["trait"][False] / total_trait) * 1
        

if __name__ == "__main__":
    main()
