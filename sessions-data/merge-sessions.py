from collections import defaultdict
import csv


def merge(path_1, path_2):
    dataPerUser = defaultdict(list)
    header = []
    i = 0
    with open(path_1, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                header.extend(row)
                i += 1
            else:
                dataPerUser[row[0]] = row
    i = 0
    with open(path_2, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            if i == 0:
                header.extend(row[1:])
                i += 1
            else:
                dataPerUser[row[0]].extend(row[1:])
    with open("sessions-engineered.csv", "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        filewriter.writerow(header)
        for id in dataPerUser:
            filewriter.writerow(dataPerUser[id])


merge("sessions_distinct_and_time_features.csv", "sessions-ratios-and-casted.csv")
