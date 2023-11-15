import argparse
import gzip
import random
import json

# def train_test_split(lang_dict, train_ratio, random_seed, use_min, train_max_size = 500, test_max_size=40):
#     print(f"Use min: {use_min}")
#     train_set = {}
#     test_set = {}
#     # test_max_size = int(max_size * (1 - train_ratio))
#     min_subdoc_num = get_min_subdocs(lang_dict)
#     for k, v in lang_dict.items():
#         subdoc_num = min_subdoc_num if use_min else len(v)
#         train_size = int(train_ratio * subdoc_num)
#         if train_size < 5:
#             continue
#         train_set[k] = []
#         test_set[k] = []
#         random.shuffle(v)
#         for i in range(subdoc_num):
#             htid = v[i][0]
#             subdoc = v[i][1]
#             if i < train_size:
#                 train_set[k].append((htid, subdoc))
#             else:
#                 test_set[k].append((htid, subdoc))
#         random.shuffle(train_set[k])
#         random.shuffle(test_set[k])
#         train_set[k] = train_set[k][:train_max_size]
#         test_set[k] = test_set[k][:test_max_size]
#         print(f"Individual train size: {len(train_set[k])}")
#         print(f"Individual test size: {len(test_set[k])}")
#     return train_set, test_set


#def get_min_subdocs(lang_dict):
#    return min(len(subdocs) for subdocs in lang_dict.values())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None)
    parser.add_argument("--ratios", dest="ratios", type = float, help = "Fractions of data for each output file")
    parser.add_argument("--outputs", dest="outputs", nargs = 2, help= "Output file names")
    args, rest = parser.parse_known_args()

    if args.random_seed:
        random.seed(args.random_seed)
    
    indices = []
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            indices.append(i)
    random.shuffle(indices)

    
    # lang_dict = {}
    # with gzip.open(args.input, "rt") as ifd:
    #     for line in ifd:
    #         data = json.loads(line)
    #         label = data['label']
    #         htid = data['htid']
    #         processed_content = (data['content']).lower().strip()
    #         if label not in lang_dict:
    #             lang_dict[label] = []
    #         lang_dict[label].append((htid, processed_content))

    # print(args.outputs)


    # # for k, v in lang_dict.items():
    # #     if not v:
    # #         print("Empty key: ", k)

    # assert(args.ratio <= 1.0)
    # assert(args.ratio >= 0.0)
    
    # train, test = train_test_split(lang_dict, args.ratio, args.random_seed, args.min != 0)
    # with gzip.open(args.outputs[0], "wt") as train_output:
    #     train_output.write(json.dumps(train))
    
    # with gzip.open(args.outputs[1], "wt") as test_output:
    #     test_output.write(json.dumps(test))

