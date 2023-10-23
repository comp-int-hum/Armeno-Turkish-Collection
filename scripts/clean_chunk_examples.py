import gzip
import json
import argparse
# get name script with longest length
def get_longest_lang(content):
    
    longest_lang = ""
    longest_len = 0
    
    for lang in content.keys():
        lang_len = len(" ".join(content[lang]))
        if lang_len > longest_len:
            longest_lang = lang
            longest_len = lang_len
            
    # Delete me / for test
    """
    if longest_lang == "Armenian":
        print("Longest: {} ({})".format(longest_lang, longest_len))
        print("    test content: {}".format(
            "".join(content["Armenian"])[:100]
        ))
        for lang in content.keys():
            lang_len = len("".join(content[lang]))
            print("    {}: {}".format(lang, lang_len))
    """
    
    return longest_lang


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--max_doc_length", dest="max_doc_length", default=200, type=int, help="Maximum length per document in words")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()
    

ofd = {}
with gzip.open(args.input, "rt") as ifd,gzip.open(args.output,"wt") as ofd:
    for line in ifd:
        data = json.loads(line)
        lang_script = ""
        txt = []
        if data["label"] == "armeno_turkish":
            lang_script = "tur_Armenian"
            txt = (data["content"]).get("Armenian")
        else:
            longest_lang = get_longest_lang(data["content"])
            lang_script = "{}_{}".format(data['label'], longest_lang)
            txt = (data["content"][longest_lang]) # only keep script of longest length
             
        #txt = [x.strip() for x in txt]
        if txt:
            tokens = "".join(txt)
            sub_len = args.max_doc_length
            if tokens:
                for start in range(0, len(tokens), sub_len):
                    end = start + sub_len
                    sub_tokens = tokens[start:end] # using the fact python will just go to end of str
                    sub_document = "".join(sub_tokens)
                    j = {
                        "htid": data["htid"],
                        "label": lang_script,
                        "content": sub_document,
                        "subdoc_num": start // sub_len,
                    }
                    ofd.write(json.dumps(j) + "\n")
