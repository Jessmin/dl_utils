import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', dest='label_file', type=str, required=True)
parser.add_argument('--o', dest='output_dir', type=str, required=True)
args = parser.parse_args()


def read_label_txt(filename: str, output_dir: str):
    with open(filename, 'r') as f:
        txts = f.readlines()
        for res in txts:
            fname, rec = res.split('\t')
            _, fname = os.path.split(fname)
            filepath = os.path.join(output_dir, f'{fname}.txt')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if os.path.exists(filepath):
                os.remove(filepath)
            with open(filepath, 'a') as f:
                for item in json.loads(rec):
                    desc = item['transcription']
                    points = item['points']
                    line = f"{','.join([','.join(str(m) for m in x) for x in points])},{desc}\n"
                    f.write(line)


if __name__ == '__main__':
    read_label_txt(args.label_file, args.output_dir)