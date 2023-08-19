import re
import os

filesInRecord = []

# checks new files to convert
with open("RLEs/record.txt", 'r') as f:
    filesInRecord.append([file.strip() for file in f.readlines()])


filesInRecord = filesInRecord[0]
files = os.listdir('RLEs')

filesToConvert = []

print('[RECORD]')
for f in filesInRecord:
    print('   ', f)

for file in files:
    if file not in filesInRecord and '.rle' in file:
        filesToConvert.append(file)

print('\n[FILES TO CONVERT]')
for f in filesToConvert:
    print('   ', f)

###################################################################
###################################################################

for file in filesToConvert:

    patternName = file[:-4]

    print(f'\n[CONVERTING {patternName}]')

    lines = []
    with open("RLEs/" + patternName + ".rle", "r") as f:
        lines = f.readlines()

    # removes lines with comment
    lines = [x for x in lines if "#" not in x]

    # saves size line to add it to SSV
    sizeLine = lines[0]

    # sets sizes of pattern
    size = lines[0].split(',')
    size = [x.strip() for x in size]

    width = size[0].split(' ')[-1]
    height = size[1].split(' ')[-1]

    print(f'[SIZE] {width, height}')

    # creates matrix for pattern
    lines = lines[1:]
    status = ''.join(lines)
    status = status.replace('\n', '')
    status = status.split('$')

    # print('[STATUS] ', status)

    def multiply(c, mult):
        if mult == '':
            return c
        else:
            return c * int(mult)

    SSV_STRING = sizeLine

    for l in status:

        multiplier = ''
        SSV_STRING += '0. '  # <- removes problem where empty line would break renderer

        for c in l:
            toAdd = ''

            if c == 'b':
                toAdd = multiply('0. ', multiplier)
                multiplier = ''
            elif c == 'o':
                toAdd = multiply('1. ', multiplier)
                multiplier = ''
            elif c == '!':
                print('[RLE PATTERN SCANNED]')
            else:
                multiplier += c

            # adds command
            SSV_STRING += toAdd

        # adds rest of dead cells
        # deleted this part

        SSV_STRING += '\n'

    # print(SSV_STRING)

    # prints in SSV file

    with open(patternName + '.ssv', 'w') as f:
        f.write(SSV_STRING)

    with open('RLEs/record.txt', 'a') as f:
        f.write('\n' + file)

    print(f'[DONE WITH {patternName}]')
