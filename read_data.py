import csv
with open('1.csv', 'r') as csvfile:
    spamreader=csv.reader(csvfile)
    for line in spamreader:
        print(line)