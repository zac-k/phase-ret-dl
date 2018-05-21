'''
Generate cvs file for plotting.

Each column of each output file is a range of errors over the test set, each from a
different errors.txt file (and typically a different ANN configuration).
'''

import csv


def single():
    path = './data/figures/details/'
    prefix = 'test_'
    ext = '.txt'
    n_samples = 100

    sources = ['Rotation', 'Scale', 'Shift', 'Noise', 'Defocus', 'Potential', 'Imaginary']
    for j, source in enumerate(sources):
        vars()[source] = []

    tie_error = []
    ann_error = []

    for i in range(0, n_samples):
        file = path + prefix + str(i) + ext
        with open(file, 'r') as csvinput:
            reader = csv.reader(csvinput, delimiter=' ')
            for j in range(len(sources)):
                next(reader)
            tie_error.append(next(reader)[-1][:-1])
            ann_error.append(next(reader)[-1][:-1])


        for source in sources:
            with open(file, 'r') as csvinput:
                reader = csv.reader(csvinput, delimiter=' ')

                for j, row in enumerate(reader):
                    if source+':' in row and 'NA' not in row:
                        x = row[2][:-1]

                        vars()[source].append(x)
    errors = []
    for source in sources:
        if vars()[source] != []:
            row = [source] + vars()[source]
            errors.append(row)
    errors.append(['tie']+tie_error)
    errors.append(['ann']+ann_error)

    errors = list(map(list, zip(*errors)))
    with open('errors_single.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(errors)


def multiple():

    # List of input filenames
    filenames = ['errors0.txt','errors20.txt','errors30.txt', 'errors40.txt', 'errors60.txt', 'errors80.txt', 'errors100.txt']

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
    with open('output_TIE.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(tie)
    with open('output_ANN.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(ann)

single()

