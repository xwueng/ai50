import csv
import sys
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    data = load_data(sys.argv[1])
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

    data = pd.read_csv(filename, delimiter=',')
    data['Revenue'] = data['Revenue'] * 1
    dv = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}
    data.VisitorType = data.VisitorType.map(dv)
    data['Weekend'] = data['Weekend'] * 1
    dm = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3,
         'May': 4, 'June': 5, 'Jul': 6,
         'Aug': 7, 'Sep': 8, 'Oct': 9,
         'Nov': 10, 'Dec': 11}
    data.Month = data.Month.map(dm)
    data_list = np.array(data).tolist()
    evidence = list()
    labels = list()
    for d in data_list:
        evidence.append(d[:-1])
        labels.append(d[-1])

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    model = neigh.fit(evidence, labels)
    return model


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

    preds = predictions.tolist()
    ntrue_pos = sum(labels)
    ntrue_neg = len(labels) - ntrue_pos
    ncorrect_pos_preds = 0
    ncorrect_neg_preds = 0
    for lab, pred in zip(labels, preds):
        ncorrect_pos_preds += (lab == 1) and (pred == 1)
        ncorrect_neg_preds += (lab == 0) and (pred == 0)
    total_correct_preds = ncorrect_pos_preds + ncorrect_neg_preds
    sensitivity = (ncorrect_pos_preds / ntrue_pos)
    specificity = (ncorrect_neg_preds / ntrue_neg)
    # print("---evaluate---")
    # print(f"len(labels): {len(labels)}: len(predictions): {len(predictions)}")
    # print(f"ncorrect_pos_preds: {ncorrect_pos_preds}")
    # print(f"ncorrect_neg_preds: {ncorrect_neg_preds}")
    # print(f"total correct preds: {total_correct_preds}")
    # print(f"total Incorrect preds: {len(predictions) - total_correct_preds}")
    # print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    # print(f"True Negative Rate: {100 * specificity:.2f}%")
    # print(f"sensitivity: {sensitivity: 2f}, specificity: {specificity: 2f}")
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
