import gzip
import json
import argparse

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
        if data["label"] == "armeno_turkish":
            lang_script = "tur_Armenian"
            txt = (data["content"]).get("Armenian")
        else:
            longest_lang = get_longest_lang(data["content"])
            lang_script = "{}_{}".format(data['label'], longest_lang)
            txt = " ".join(data["content"][longest_lang])
             
        #txt = [x.strip() for x in txt]
        if txt:
            tokens = " ".join(txt)
            sub_len = args.max_doc_length
            if tokens:
                num_subdocs = int(len(tokens)/sub_len)
                for subnum in range(num_subdocs):
                    start = subnum * sub_len
                    end = (subnum+1) * sub_len
                    sub_tokens = tokens[start:end]
                    sub_document = "".join(sub_tokens)
                ofd.write(json.dumps(
                {
                    "htid" : data["htid"],
                    "label" : lang_script,
                    "content" : sub_document,
                }
                ))
        
