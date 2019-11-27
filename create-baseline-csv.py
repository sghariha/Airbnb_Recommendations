import csv
from collections import defaultdict
import datetime

def readBaselineData(path):
    data = []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        i = 0
        for row in reader:
            if i == 0:
                data.append(
                    [
                        row[0],
                        row[1],
                        row[2],
                        row[4],
                        row[5],
                        row[7],
                        row[8],
                        row[9],
                        row[10],
                        row[12],
                        row[13],
                        row[14],
                        row[15],
                    ]
                )
                i += 1
            elif row[5] != "":
                # do not add index 3 (first date booked), 6 (signup method)
                featureVector = []
                featureVector.extend(
                    [
                        row[0],
                        row[1],
                        row[2],
                        row[4],
                        float(row[5]),
                        float(row[7]),
                        row[8],
                        row[9],
                        row[10],
                        row[12],
                        row[13],
                        row[14],
                        row[15],
                    ]
                )
                data.append(featureVector)
    return data


def createBaselineCSV(data):
    csvContent = []
    titleVector = []
    genderValues = []
    languageValues = []
    affiliateChannelValues = []
    affiliateProviderValues = []
    signupAppValues = []
    firstDeviceTypeValues = []
    firstBrowserValues = []
    for i in range(0, len(data[0])):
        # age is now col index 4, signup flow is now col index 5 in data matrix etc.
        if i == 0 or i == 4 or i == 5:
            titleVector.extend([data[0][i]])
        elif i == 1 or i == 2:
            titleVector.extend(createTitlesForDates(data[0][i], 2009, 2014))
        else:
            col = []
            for j in range(1, len(data)):
                col.append(data[j][i])
            uniqueValues = list(extractUniqueValues(col))
            if i == 3:
                genderValues = uniqueValues
            elif i == 6:
                languageValues = uniqueValues
            elif i == 7:
                affiliateChannelValues = uniqueValues
            elif i == 8:
                affiliateProviderValues = uniqueValues
            elif i == 9:
                signupAppValues = uniqueValues
            elif i == 10:
                firstDeviceTypeValues = uniqueValues
            else:
                firstBrowserValues = uniqueValues
            titleVector.extend(createTitlesForNonDates(data[0][i], uniqueValues))
    csvContent.append(titleVector)
    for i in range(1, len(data)):
        featureVector = []
        for j in range(0, len(data[0])):
            if j == 0 or j == 4 or j == 5:
                featureVector.extend([data[i][j]])
            elif j == 1 or j == 2:
                featureVector.extend(oneHotEncodeDate(data[i][j], 2009, 2014))
            else:
                if j == 3:
                    featureVector.extend(oneHotEncodeNonDate(genderValues, data[i][j]))
                elif j == 6:
                    featureVector.extend(
                        oneHotEncodeNonDate(languageValues, data[i][j])
                    )
                elif j == 7:
                    featureVector.extend(
                        oneHotEncodeNonDate(affiliateChannelValues, data[i][j])
                    )
                elif j == 8:
                    featureVector.extend(
                        oneHotEncodeNonDate(affiliateProviderValues, data[i][j])
                    )
                elif j == 9:
                    featureVector.extend(
                        oneHotEncodeNonDate(signupAppValues, data[i][j])
                    )
                elif j == 10:
                    featureVector.extend(
                        oneHotEncodeNonDate(firstDeviceTypeValues, data[i][j])
                    )
                else:
                    featureVector.extend(
                        oneHotEncodeNonDate(firstBrowserValues, data[i][j])
                    )
        csvContent.append(featureVector)
    with open("baseline.csv", "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in csvContent:
            filewriter.writerow(row)


def oneHotEncodeNonDate(uniqueValues, value):
    nonDateEncoding = [0] * len(uniqueValues)
    for i in range(0, len(uniqueValues)):
        if uniqueValues[i] == value:
            nonDateEncoding[i] = 1
    return nonDateEncoding


def oneHotEncodeDate(date, startYear, endYear):
    oneHotEncoding = []
    year = int()
    month = int()
    day = int()
    if "-" in date:
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
    else:
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:8])

    yearEncoding = [0] * (endYear - startYear + 1)
    yearEncoding[year - startYear] = 1
    monthEncoding = [0] * (12)
    monthEncoding[month - 1] = 1
    dayEncoding = [0] * (31)
    dayEncoding[day - 1] = 1
    date = datetime.datetime(year, month, day)
    DOFEncoding = [0] * 7
    # Monday is index 0,...,Sunday is index 6
    DOFEncoding[date.weekday()] = 1

    oneHotEncoding.extend(yearEncoding)
    oneHotEncoding.extend(monthEncoding)
    oneHotEncoding.extend(dayEncoding)
    oneHotEncoding.extend(DOFEncoding)

    return oneHotEncoding


def extractUniqueValues(col):
    values = set()
    for item in col[1:]:
        values.add(item)
    return values


def createTitlesForNonDates(featureName, uniqueValues):
    titleVector = [featureName + " "] * len(uniqueValues)
    for i in range(0, len(uniqueValues)):
        titleVector[i] += uniqueValues[i]
    return titleVector


def createTitlesForDates(featureName, startYear, endYear):
    years = [featureName + " "] * (endYear - startYear + 1)
    for i in range(startYear, endYear + 1):
        years[i - startYear] += str(i)
    months = [
        featureName + " jan",
        featureName + " feb",
        featureName + " mar",
        featureName + " apr",
        featureName + " may",
        featureName + " jun",
        featureName + " jul",
        featureName + " aug",
        featureName + " sep",
        featureName + " oct",
        featureName + " nov",
        featureName + " dec",
    ]
    days = [featureName + " day"] * 31
    for i in range(1, 32):
        days[i - 1] += str(i)
    daysOfWeek = [
        featureName + " mon",
        featureName + " tue",
        featureName + " wed",
        featureName + " thu",
        featureName + " fri",
        featureName + " sat",
        featureName + " sun",
    ]
    titleVector = []
    titleVector.extend(years)
    titleVector.extend(months)
    titleVector.extend(days)
    titleVector.extend(daysOfWeek)
    return titleVector


createBaselineCSV(readBaselineData("train_users_2.csv"))
