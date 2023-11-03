import argparse
import gzip
import random
import json

def train_test_split(lang_dict, train_ratio, random_seed, use_min, max_size = 1):
    random.seed(random_seed)
    print(f"Use min: {use_min}")
    train_set = {}
    test_set = {}
    train_max_size = max_size
    # test_max_size = int(max_size * (1 - train_ratio))
    min_subdoc_num = get_min_subdocs(lang_dict)
    for k, v in lang_dict.items():
        subdoc_num = min_subdoc_num if use_min else len(v)
        train_size = int(train_ratio * subdoc_num)
        if train_size < 5:
            continue
        train_set[k] = []
        test_set[k] = []
        random.shuffle(v)
        for i in range(subdoc_num):
            htid = v[i][0]
            subdoc = v[i][1]
            if i < train_size:
                train_set[k].append((htid, subdoc))
            else:
                test_set[k].append((htid, subdoc))
        random.shuffle(train_set[k])
        random.shuffle(test_set[k])
        train_set[k] = train_set[k][:train_max_size]
    print("Overall train size", len(train_set))
    print("Overall test size", len(test_set))
    return train_set, test_set


def get_min_subdocs(lang_dict):
    return min(len(subdocs) for subdocs in lang_dict.values())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        Takes an input data file, a list of output file names, and a corresponding list of proportions.
        Shuffles the input lines, and splits them into the output files according to the proportions.

        Note: does not write the actual input lines to the output files, but instead writes the *line numbers*.
        This avoids duplicating large amounts of data, but means that the next script will need both the
        random split file *and* the original data file.

        All inputs/outputs are read/written with gzip compression.  A random seed may be specified for
        reproducibility.
        """
        )
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None)
    parser.add_argument("--use_min", dest="min", type=int, default=0)
    parser.add_argument("--train_ratio", dest="ratio", type = float, help = "fraction of data for training")
    parser.add_argument("--outputs", dest="outputs", nargs = 2, help= "names for train and test files")
    args, rest = parser.parse_known_args()
    
    lang_dict = {}
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            data = json.loads(line)
            label = data['label']
            htid = data['htid']
            processed_content = (data['content']).lower().strip()
            if label not in lang_dict:
                lang_dict[label] = []
            lang_dict[label].append((htid, processed_content))

    print(args.outputs)


    # for k, v in lang_dict.items():
    #     if not v:
    #         print("Empty key: ", k)

    assert(args.ratio <= 1.0)
    assert(args.ratio >= 0.0)
    
    train, test = train_test_split(lang_dict, args.ratio, args.random_seed, args.min != 0)
    with gzip.open(args.outputs[0], "wt") as train_output:
        train_output.write(json.dumps(train))
    
    with gzip.open(args.outputs[1], "wt") as test_output:
        test_output.write(json.dumps(test))

