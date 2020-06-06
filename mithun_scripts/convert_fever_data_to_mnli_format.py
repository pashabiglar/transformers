import pandas as pd
import csv
import os
import argparse


#--file_path "../src/transformers/data/datasets/fever/feverindomain/lex/train.tsv"

def create_parser():
    parser = argparse.ArgumentParser(description='convert to mnli')
    parser.add_argument('--file_path', default="error.tsv", type=str,
                        help='path to input file')
    return parser

def parse_commandline_args():
    return create_parser().parse_args()


def reader(input_filepath):
    with open(input_filepath,"r",encoding="utf-8-sig") as filepath:
        input_data=pd.read_csv(filepath,"\t",header=None)
        return input_data.values.tolist()

def writer(data,file_path):
    header_mnli=['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']
    with open(file_path,"w") as filepath:
        mnli_writer=csv.writer(filepath,delimiter="\t")
        mnli_writer.writerow(header_mnli)
        for index,row in enumerate(data):
            mnli_format=[index, 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', row[2], row[3], row[1], row[1]]
            mnli_writer.writerow(mnli_format)
    filepath.close()


args = parse_commandline_args()
if not (os.path.exists(args.file_path)):
    raise ValueError(
        f"input file ({args.file_path}) doesn't exist")

read_data=reader(args.file_path)
write_data=writer(read_data,args.file_path)
