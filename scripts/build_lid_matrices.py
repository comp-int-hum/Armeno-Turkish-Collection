import json
import gzip
import logging
import argparse
import numpy as np
logger = logging.getLogger("doc_id")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        
        for line in ifd:
            j = json.loads(line)
            logger.info("ID'ing '%s'", j["htid"])
            key_list = j["lid"][0].keys()
            k_len = len(key_list)
            assert all(len(subdoc.keys()) == k_len for subdoc in j["lid"])
            full_dist_list = list(map(lambda x: x.items(), j["lid"]))
            # full_dist_keys = list(map(lambda x: x.keys(), j["lid"])) # for debugging purposes
            full_dist_matrix = np.array(full_dist_list).T # transpose so that lid is dim = 0
            cleaned_fd = np.nan_to_num(full_dist_matrix) # clean
            total_prob = np.sum(cleaned_fd, axis=1) # sum along dim = 1
            max_index = np.argmax(total_prob)
            fd_list = cleaned_fd.tolist()
            doc = {
                "htid": j["htid"],
                "max_lid_dist": fd_list[max_index],
                "full_lid_dist": fd_list, # outer list is language id, inner list is time
                "key_list": key_list,
                "max_lang": key_list[max_index]
            }

            ofd.write(json.dumps(doc) + "\n")
            
            
            # logger.info(f"{j.keys()}")
            # logger.info(f"{j['lid'][0].keys()}")
            # logger.info(f"{j['lid'][0]['lit_Latin']}")
            break
