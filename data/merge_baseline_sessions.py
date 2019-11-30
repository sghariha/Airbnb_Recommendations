from collections import defaultdict
import csv

# pass in relative path to Kevin's baseline csv and Alex + Shreeman's merged sessions csv
def mergeBaselineAndSessionFeatures(baselinePath, sessionPath, outputCSVPath):
    baselineData = defaultdict(list)
    sessionData = defaultdict(list)
    header = []
    sessionColumns = 0
    trainingUsersWithSessions = 0

    # gather data per user in baseline csv and create csv header
    i = 0
    with open(baselinePath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                header.extend(row)
                i += 1
            else:
                baselineData[row[2]] = row

    # gather data per user in sessions csv and add to csv header
    i = 0
    with open(sessionPath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                header.extend(row[1:])
                sessionColumns = len(row) - 1
                i += 1
            else:
                sessionData[row[0]] = row

    # to hold averaged session data across all users in baseline that have session data
    averageSessionFeatures = [0.0] * sessionColumns

    # extend baseline data to contain users' session data and compute average vector
    for user in baselineData:
        if user in sessionData.keys():
            baselineData[user].extend(sessionData[user][1:])
            trainingUsersWithSessions += 1
            dataToAppend = sessionData[user][1:]
            averageSessionFeatures = [
                averageSessionFeatures[i] + float(dataToAppend[i])
                for i in range(sessionColumns)
            ]
    averageSessionFeatures = [
        averageSessionFeatures[i] / float(trainingUsersWithSessions)
        for i in range(sessionColumns)
    ]

    # extend baseline data to hold average session data for users w/o session data
    for user in baselineData:
        if user not in sessionData.keys():
            baselineData[user].extend(averageSessionFeatures)

    # write to csv
    with open(outputCSVPath, "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        filewriter.writerow(header)
        for id in baselineData:
            filewriter.writerow(baselineData[id])


mergeBaselineAndSessionFeatures(
    "test_users-processed.csv", "sessions-engineered.csv", "feature-matrix.csv"
)
