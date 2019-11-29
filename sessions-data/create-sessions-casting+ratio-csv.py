from collections import defaultdict
import csv
import math
import pandas as pd

df = pd.read_csv("sessions.csv")
uniqueActionsOverall = df.action.unique()
uniqueActionTypesOverall = df.action_type.unique()
uniqueActionDetailsOverall = df.action_detail.unique()
uniqueDeviceTypesOverall = df.device_type.unique()
del df


def readCSVIntoMap(path):
    userSessionsMap = defaultdict(list)

    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i = 0
        for row in reader:
            if i == 0:
                i += 1
            else:
                if not (type(row[0]) == float and math.isnan(row[0])):
                    userSessionsMap[row[0]].append(row)
    return userSessionsMap


def createHeader():
    headerVector = []
    headerVector.extend(
        [
            "id",
            "ratio_distinct_actions",
            "ratio_distinct_actions_types",
            "ratio_distinct_action_details",
            "ratio_distinct_devices",
        ]
    )
    for action in uniqueActionsOverall:
        if not (type(action) == float and math.isnan(action)):
            headerVector.extend(
                ["casted_action_" + str(action).strip().replace(" ", "") + "_ratio"]
            )
    for actionType in uniqueActionTypesOverall:
        if not (type(actionType) == float and math.isnan(actionType)):
            headerVector.extend(
                [
                    "casted_action_type_"
                    + str(actionType).strip().replace(" ", "")
                    + "_ratio"
                ]
            )
    for actionDetail in uniqueActionDetailsOverall:
        if not (type(actionDetail) == float and math.isnan(actionDetail)):
            headerVector.extend(
                [
                    "casted_action_detail_"
                    + str(actionDetail).strip().replace(" ", "")
                    + "_ratio"
                ]
            )
    for deviceType in uniqueDeviceTypesOverall:
        if not (type(deviceType) == float and math.isnan(deviceType)):
            headerVector.extend(
                [
                    "casted_device_type_"
                    + str(deviceType).strip().replace(" ", "")
                    + "_ratio"
                ]
            )
    return headerVector


def computeStats(map):
    results = []
    results.append(createHeader())

    for id in map:
        featureVector = []
        uniqueActionsForUser = defaultdict(int)
        uniqueActionTypesForUser = defaultdict(int)
        uniqueActionDetailsForUser = defaultdict(int)
        uniqueDeviceTypesForUser = defaultdict(int)
        numSessionsForUser = len(map[id])
        for session in map[id]:
            uniqueActionsForUser[session[1]] += 1
            uniqueActionTypesForUser[session[2]] += 1
            uniqueActionDetailsForUser[session[3]] += 1
            uniqueDeviceTypesForUser[session[4]] += 1
        featureVector.extend(
            [
                id,
                len(uniqueActionsForUser.keys()) / len(uniqueActionsOverall),
                len(uniqueActionTypesForUser.keys()) / len(uniqueActionTypesOverall),
                len(uniqueActionDetailsForUser.keys())
                / len(uniqueActionDetailsOverall),
                len(uniqueDeviceTypesForUser.keys()) / len(uniqueDeviceTypesOverall),
            ]
        )
        for action in uniqueActionsOverall:
            if not (type(action) == float and math.isnan(action)):
                featureVector.extend(
                    [uniqueActionsForUser[action] / numSessionsForUser]
                )

        for actionType in uniqueActionTypesOverall:
            if not (type(actionType) == float and math.isnan(actionType)):
                featureVector.extend(
                    [uniqueActionTypesForUser[actionType] / numSessionsForUser]
                )

        for actionDetail in uniqueActionDetailsOverall:
            if not (type(actionDetail) == float and math.isnan(actionDetail)):
                featureVector.extend(
                    [uniqueActionDetailsForUser[actionDetail] / numSessionsForUser]
                )

        for deviceTypes in uniqueDeviceTypesOverall:
            if not (type(deviceTypes) == float and math.isnan(deviceTypes)):
                featureVector.extend(
                    [uniqueDeviceTypesForUser[deviceTypes] / numSessionsForUser]
                )
        results.append(featureVector)
    return results


def writeDataIntoCSV(data):
    with open("sessions-ratios-and-casted.csv", "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in data:
            filewriter.writerow(row)


dataMap = readCSVIntoMap("sessions.csv")
toWrite = computeStats(dataMap)
writeDataIntoCSV(toWrite)
