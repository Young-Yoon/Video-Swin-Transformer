import csv
modes = ['train', 'test', 'val']
parts = ['0', 'aa']
fstarts = [2, 0]
mode, part, fstart = modes[2], parts[1], fstarts[1]
f_in = f'{mode}_{part}'   # 'test_0'
f_label = 'label_map_k400.txt'
f_ann = f'../data/annotations/{mode}.csv'
f_list = '../data/targz_list/'+f_in+'.txt'
f_list_label = f_list[:-3] + 'ann'

with open(f_label, 'r') as f:
    label2int = {row.strip(): i for i, row in enumerate(f.readlines())}

file2ann = {}
with open(f_ann, 'r') as f:
    # y = , delimiter=',')
    for i, row in enumerate(csv.reader(f)):
        row[0] = label2int[row[0]] if row[0] in label2int else -1
        row[1] = '_'.join([row[1], row[2].rjust(6, '0'), row[3].rjust(6, '0')])+'.mp4'
        file2ann[row[1]] = row[0]

fout = open(f_list_label, 'w')
with open(f_list, 'r') as f:
    for i, row in enumerate(f.readlines()):
        fname = row.strip()
        print(fname, file2ann[fname[fstart:]])
        fout.writelines(' '.join([fname, str(file2ann[fname[fstart:]])])+'\n')
fout.close()