import re
import argparse
import gzip
import json
import csv
import random
import os.path
import unicodedata
import zipfile
from pairtree import PairtreeStorageFactory
#import fasttext
from pymarc.reader import JSONReader
from pymarc.writer import XMLWriter
from pymarc.record import Record

from huggingface_hub import hf_hub_download

#model_path = hf_hub_download(repo_id="arc-r/fasttext-language-identification", filename="lid.176.bin")
#model = fasttext.load_model(model_path)
#labels, probs = model.predict(chunks, k=3)
    
csv.field_size_limit(1000000000)

# TODO: split each document by script and rerun LID, sample over time

def get_info(obj):
    htid, author, title, author_date, lang, pub_place, pub_date = None, None, None, None, None, None, None
    for item in obj["fields"]:
        if "974" in item.keys():
            for subf in item["974"]["subfields"]:
                var_keys = subf.keys()
                if 'u' in var_keys:
                    htid = subf['u']
        elif "008" in item.keys():
            lang = item["008"][35:38]
            pub_date = item["008"][7:11]
            pub_place = item["008"][15:18]
        elif "100" in item.keys():
            for subf in item["100"]["subfields"]:
                if "a" in subf.keys():
                    author = subf["a"]
                elif "d" in subf.keys():
                    author_date = subf["d"]
        elif "245" in item.keys():
            for subf in item["245"]["subfields"]:
                if "a" in subf.keys():
                    title = subf["a"]
    return (htid, author, title, author_date, lang, pub_place, pub_date)


def get_buckets(dates, num=5):
    min_date = min(dates)
    max_date = max(dates)
    date_range = max_date - min_date
    window_size = date_range / num
    buckets = [[] for i in range(num)]
    for i, date in enumerate(dates):
        for j in range(num):
            if date <= min_date + j * window_size:
                buckets[j].append(i)
                break
    return buckets



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hathitrust_root", dest="hathitrust_root", help="Path to Hathi Trust")
    parser.add_argument("--unicode_scripts", dest="unicode_scripts", default="data/Scripts.txt", help="Unicode Script.txt file")
    parser.add_argument("--per_language", dest="per_language", default=10, type=int, help="Number of documents per language")
    parser.add_argument("--min_total_per_language", dest="min_total_per_language", default=100, type=int, help="Number of total documents available per language")
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()
    
    if args.random_seed:
        random.seed(args.random_seed)

    psf = PairtreeStorageFactory()        
    def process_doc(htid, language, title, author, author_date, pub_place, pub_date, ofd):
        subcollection, ident = re.match(r"^([^\.]+)\.(.*)$", htid).groups()
        try:
            store = psf.get_store(
                store_dir=os.path.join(
                    args.hathitrust_root,
                    subcollection
                )
            )
        except:
            return False
        try:
            obj = store.get_object(ident, create_if_doesnt_exist=False)
        except:                
            return False    
        pages = []
        for subpath in obj.list_parts():
            for fname in obj.list_parts(subpath):
                if fname.endswith("zip"):
                    with zipfile.ZipFile(
                            obj.get_bytestream(
                                "{}/{}".format(subpath, fname),
                                streamable=True
                            )
                    ) as izf:                            
                        for page in sorted(izf.namelist()):
                            if page.endswith("txt"):
                                txt = izf.read(page).decode("utf-8")
                                pages.append(txt)

        txt = "\n".join(pages)
        prev_script = None
        cur = ""
        scripts = {}
        #script_counts = {}
        for c in txt:
            s = script_lookup.get(ord(c), "Common")
            if s != "Common":
                if prev_script in [None, s]:
                    cur += c
                else:
                    if len(cur) > 0:
                        scripts[prev_script] = scripts.get(prev_script, [])
                        scripts[prev_script].append(cur)
                        cur = ""
                prev_script = s
            else:
                cur += c
        if len(cur) > 0:
            scripts[prev_script] = scripts.get(prev_script, [])
            scripts[prev_script].append(cur)                    

        if len(scripts) > 0:
            ofd.write(json.dumps(
                {
                    "language" : language,
                    "title" : title,
                    "author" : author,
                    "scripts" : scripts,
                    "htid" : htid,
                    "author_date" : author_date,
                    "pub_date" : pub_date,
                    "pub_place" : pub_place
                }
            ) + "\n")
            return True
        else:
            return False


        
    # create a dictionary from unicode integer to script
    script_lookup = {}
    with open(args.unicode_scripts, "rt") as ifd:
        for line in ifd:
            m = re.match(r"^(\S+?)(?:\.\.(\S+))?\s+;\s+([^\#]+)\s+\#.*$", line)
            if m:
                start, end, script = m.groups()
                start = int(start, base=16)
                for i in range(start, 1 + (start if not end else int(end, base=16))):
                    script_lookup[i] = script

    # create a dictionary from language code to lists of documents with that code 
    langs = {}
    with gzip.open(os.path.join(args.hathitrust_root, "full_marc.json.gz"), "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            htid, author, title, author_date, lang, pub_place, pub_date = get_info(j)
            try:
                pub_date = int(re.sub(r"\D", "", pub_date))
            except:
                continue
            if htid and pub_date > 1500:                
                langs[lang] = langs.get(lang, [])
                langs[lang].append((htid, author, title, author_date, pub_place, pub_date))

    langs = {k : v for k, v in langs.items() if re.match(r"[a-z]{3}", k) and len(v) >= args.min_total_per_language}

    
    with gzip.open(args.output, "wt") as ofd:
        for lang, docs in langs.items():
            dates = [d[5] for d in docs]
            rest = []
            count = 0
            for bucket in get_buckets(dates):
                random.shuffle(bucket)
                for i in range(len(bucket)):
                    htid, author, title, author_date, pub_place, pub_date = docs[bucket[i]]
                    suc = process_doc(htid, lang, title, author, author_date, pub_place, pub_date, ofd)
                    if suc:
                        count += 1
                        rest += bucket[i + 1:]
                        break
            random.shuffle(rest)
            for i in rest:
                if count >= args.per_language:
                    break
                htid, author, title, author_date, pub_place, pub_date = docs[i]
                suc = process_doc(htid, lang, title, author, author_date, pub_place, pub_date, ofd)
                print(suc, count, i)
                if suc:
                    count += 1
