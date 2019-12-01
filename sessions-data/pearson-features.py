from scipy.stats import pearsonr
import csv
from collections import defaultdict

# input - pandas dataframe for training or test data
def pearsonCorrelation(path):
    dataPerUser = defaultdict(list)

    actionsPearsonVectorsPerUser = []
    actionsPearsonCoefficientFeature = []

    actionDetailsPearsonVectorsPerUser = []
    actionDetailsPearsonCoefficientFeature = []

    actionsColumns = []
    actionDetailsColumns = []

    i = 0
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                j = 0
                for column in row:
                    if (
                        "casted_action_" in column
                        and "casted_action_detail_" not in column
                    ):
                        actionsColumns.append(j)
                    elif "casted_action_detail_" in column:
                        actionDetailsColumns.append(j)
                    j += 1
                i += 1
            else:
                dataPerUser[row[2]] = row

    print(actionsColumns)
    print(actionDetailsColumns)

    i = 0
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                i += 1
                continue
            actionsPearsonVector = []
            actionDetailsPearsonVector = []
            for j in actionsColumns:
                actionsPearsonVector.append(float(row[j]))
            for j in actionDetailsColumns:
                actionDetailsPearsonVector.append(float(row[j]))
            actionsPearsonVectorsPerUser.append(actionsPearsonVector)
            actionDetailsPearsonVectorsPerUser.append(actionDetailsPearsonVector)

    numUsers = len(actionsPearsonVectorsPerUser)
    for i in range(numUsers):
        maxActionsCorrelation = float("-inf")
        maxActionDetailsCorrelation = float("-inf")
        for j in range(numUsers):
            if i != j:
                actionsCorrelation, _ = pearsonr(
                    actionsPearsonVectorsPerUser[i], actionsPearsonVectorsPerUser[j]
                )
                actionDetailsCorrelation, _ = pearsonr(
                    actionDetailsPearsonVectorsPerUser[i],
                    actionDetailsPearsonVectorsPerUser[j],
                )
                if actionsCorrelation > maxActionsCorrelation:
                    maxActionsCorrelation = actionsCorrelation
                if actionDetailsCorrelation > maxActionDetailsCorrelation:
                    maxActionDetailsCorrelation = actionDetailsCorrelation
        actionsPearsonCoefficientFeature.append(maxActionsCorrelation)
        actionDetailsPearsonCoefficientFeature.append(maxActionDetailsCorrelation)

        print(len(actionsPearsonCoefficientFeature))
        print(len(actionDetailsPearsonCoefficientFeature))


pearsonCorrelation("sessions-engineered.csv")
