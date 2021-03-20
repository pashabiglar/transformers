#this was because when sandeep did figer conversion he somehow messed up line numbers..rather at each line he stored
# the actual line number of the coreresponding data point in lex, as a value called index.
'''
This code is used to match the delex data with lex counter part in the combined/student teacher architecture.
had to do this because th TSV files sandeep created for figer* didnt match with lex order.
input=FILE_TO_CHANGE: give the input path to the wrongly encoded file
output=
'''

FILE_TO_CHANGE="/Users/mordor/Downloads/cross_domain_train2.tsv"
import pandas
import csv
from tqdm import tqdm

def _read_tsv( input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

def _create_examples( lines, set_type):
    """Creates examples for the training, dev and test sets."""
    all_inputs_with_index=[]
    all_inputs_rearranged=[None]*len(lines)
    for (i, line) in tqdm(enumerate(lines), desc="creating examples", total=len(lines)):
        index = line[4]
        index_int=int(index.split("-")[1])
        all_inputs_rearranged[index_int-1]=line
    return all_inputs_rearranged

def write_csv(all_inputs_rearranged):
    import csv
    with open('fever_test_figerabs_index_fixed.tsv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for line in all_inputs_rearranged:
            spamwriter.writerow(line)
    csvfile.close()
    return

all_inputs_rearranged=_create_examples(_read_tsv(FILE_TO_CHANGE),"dev")
write_csv(all_inputs_rearranged)