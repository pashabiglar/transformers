import pandas as pd
import csv

input_filepath="/Users/mordor/research/huggingface/src/transformers/data/datasets/FeverInDomain/dev_sandeep_bert_format.tsv"
output_filepath="/Users/mordor/research/huggingface/src/transformers/data/datasets/FeverInDomain/dev_mnli_format.tsv"

def reader(input_filepath):
    with open(input_filepath,"r",encoding="utf-8-sig") as filepath:
        input_data=pd.read_csv(filepath,"\t",header=None)
        return input_data.values.tolist()

def writer(data):
    header_mnli=['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']
    with open(output_filepath,"w") as filepath:
        mnli_writer=csv.writer(filepath,delimiter="\t")
        mnli_writer.writerow(header_mnli)
        for index,row in enumerate(data):
            mnli_format=[index, 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', row[2], row[3], row[1], row[1]]
            mnli_writer.writerow(mnli_format)
    filepath.close()


read_data=reader(input_filepath)
write_data=writer(read_data)