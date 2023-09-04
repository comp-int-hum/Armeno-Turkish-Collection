import json
import gzip
import argparse
import math
import fasttext
from huggingface_hub import hf_hub_download


#labels, probs = model.predict(chunks, k=3)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--window_size", dest="window_size", type=int, default=400)
    args = parser.parse_args()

    model_path = hf_hub_download(repo_id="arc-r/fasttext-language-identification", filename="lid.176.bin")
    model = fasttext.load_model(model_path)

    def process(txt):
        spans = []
        for w in range(math.ceil(len(txt) / args.window_size)):
            spans.append(txt[args.window_size * w : args.window_size * (w + 1)].replace("\n", " "))
        labels, probs = model.predict(spans, k=3)
        return [(s, list(zip(ls, [float(p) for p in ps]))) for s, ls, ps in zip(spans, labels, probs)]
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            j["content"] = {k : [process(x) for x in v] for k, v in j["content"].items()}
            ofd.write(json.dumps(j) + "\n")
