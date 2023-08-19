
values = []

with open("test.ssv", "r") as f:
    lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
        line = lines[i].split(' ')
        values.append(line)

print(values)
