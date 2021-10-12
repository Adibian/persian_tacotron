import re

files_path = "files/"

""" Read phonemes """
f = open(files_path + 'phoneme_transcriptions.txt', 'r')
lines = f.readlines()

# chars = set()

""" spliting test, train anf validation data """
test_lines = lines[2800:]
lines = lines[:2800]
val_lines = lines[:int(len(lines)/10)]
train_lines = lines[int(len(lines)/10):]

def data_to_dict(line):
    phonemes = []
    for line in line:
        line = line.replace('\n', '')
        phoneme = line.split('\t')[1]
        name = line.split('\t')[0]

        phoneme = phoneme.replace('CH', 'c').replace('KH', 'k').replace('SH', 's').replace('SIL', 'i').replace('AH', 'h').replace('ZH', 'z').replace('AA', 'a')  ## replace multi chars by one new char
        phoneme = re.sub("\[([0-9]+)\]\s*", '', phoneme)
        # chars.update(set(phoneme.split(' ')))
        phenome = "/mnt/hdd1/adibian/Tacotron2/files/wavs/" + name + ".wav|" + phoneme
        phonemes.append(phenome)
    return phonemes

train_data = data_to_dict(train_lines)
test_data = data_to_dict(test_lines)
val_data = data_to_dict(val_lines)

with open(files_path + 'text_files/train_data.txt', 'w') as fp:
    for line in train_data:
        fp.write(line + '\n')
with open(files_path + 'text_files/test_data.txt', 'w') as fp:
    for line in test_data:
        fp.write(line + '\n')
with open(files_path + 'text_files/val_data.txt', 'w') as fp:
    for line in val_data:
        fp.write(line + '\n')
