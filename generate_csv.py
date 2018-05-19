'''
Generate cvs file for plotting.

Each column of each output file is a range of errors over the test set, each from a
different errors.txt file (and typically a different ANN configuration).
'''

import csv

# List of input filenames
filenames = ['errors0.txt','errors1.txt', 'errors2.txt', 'errors5.txt', 'errors10.txt', 'errors15.txt', 'errors20.txt', 'errors25.txt', 'errors30.txt', 'errors35.txt', 'errors40.txt', 'errors45.txt', 'errors50.txt', 'errors55.txt', 'errors60.txt']

# Iterate over the files and extract the TIE and ANN errors into two
# separate lists of lists.
tie = []
ann = []
for filename in filenames:
    col = []
    with open(filename, 'r') as csvinput:
        reader = csv.reader(csvinput, delimiter=' ')

        for row in reader:
            if 'test' in row and 'input' in row:
                error = row[6]
                col.append(error[:-1])
    set_size = int(len(col)/2)
    tie.append(col[:set_size])
    ann.append(col[set_size:])

# Transpose both lists so that each column represents the data from a single input file
tie = list(map(list, zip(*tie)))
ann = list(map(list, zip(*ann)))

# Write the output of each list to a separate csv file
with open('output_TIE.csv','w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(tie)
with open('output_ANN.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(ann)



