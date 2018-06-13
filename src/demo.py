import csv

with open('../docs/demo.csv') as f:
    person = csv.reader(f)
    list = [row for row in person]
    print(list[2][1])