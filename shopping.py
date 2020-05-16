import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

# Column Index
ADMINISTRATIVE = 0
ADMINISTRATIVE_DURATION = 1
INFORMATIONAL = 2
INFORMATIONAL_DURATION = 3
PRODUCTRELATED = 4
PRODUCTRELATED_DURATION = 5
BOUNCERATES = 6
EXITRATES = 7
PAGEVALUES = 8
SPECIALDAY = 9
MONTH = 10
OPERATINGSYSTEMS = 11
BROWSER = 12
REGION = 13
TRAFFICTYPE = 14
VISITORTYPE = 15
WEEKEND = 16
REVENUE = 17

# Dictionary with months
months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, newline='') as f: 
        reader = csv.reader(f)
        next(reader) # Skip first line (headers)
        for row in reader:
            temp_list = []

            # Append all numeric values right away (as int or float) to list of evidence
            temp_list.append(int(row[ADMINISTRATIVE]))
            temp_list.append(float(row[ADMINISTRATIVE_DURATION]))
            temp_list.append(int(row[INFORMATIONAL]))
            temp_list.append(float(row[INFORMATIONAL_DURATION]))
            temp_list.append(int(row[PRODUCTRELATED]))
            temp_list.append(float(row[PRODUCTRELATED_DURATION]))
            temp_list.append(float(row[BOUNCERATES]))
            temp_list.append(float(row[EXITRATES]))
            temp_list.append(float(row[PAGEVALUES]))
            temp_list.append(float(row[SPECIALDAY]))
            
            # Convert Month to equal 0 for jan , 1 Feb etc.
            month_int = months[row[MONTH]]
            temp_list.append(month_int)

            temp_list.append(int(row[OPERATINGSYSTEMS]))
            temp_list.append(int(row[BROWSER]))
            temp_list.append(int(row[REGION]))
            temp_list.append(int(row[TRAFFICTYPE]))
           
            # Convert Visitor to equal 1 if Returning and 0 if New
            visitor_type = 1 if row[VISITORTYPE] == "Returning_Visitor" else 0
            temp_list.append(visitor_type)
            
            # Convert Weekend to equal 1 if weekend and 0 if not weekend
            weekend = 1 if row[WEEKEND] == 'TRUE' else 0
            temp_list.append(weekend)

            # Convert Revenue to equal 1 if purchase and 0 if not purchase
            revenue = 1 if row[REVENUE] == 'TRUE' else 0
            
            # Append row evidence & label to overall lists
            evidence.append(temp_list)
            labels.append(revenue)

    print(f'\n> Evidence and labels were loaded | {len(labels)} rows\n')
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors = 1)
    print('> Model was fitted | Nearest Neighbor Classifier\n')
    return neigh.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_true = 0
    positive_false = 0
    negative_true = 0
    negative_false = 0

    for n in range(len(labels)):
        
        # Count positive rate
        if labels[n] == 1:
            if labels[n] == predictions[n]:
                positive_true += 1
            else:
                positive_false += 1
        
        # Count negative rate 
        else:
            if labels[n] == predictions[n]: 
                negative_true += 1
            else: 
                negative_false += 1

    sensitivity = float(positive_true / (positive_true + positive_false))
    specificity = float(negative_true / (negative_true + negative_false))

    print("> Sensitivity and specificity were calculated.\n")
    return(sensitivity, specificity)


if __name__ == "__main__":
    main()
