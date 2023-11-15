import argparse
import gzip
import zipfile
import json
import os.path
import re
from pairtree import PairtreeStorageFactory

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--split_by_script", dest="split_by_script", default=False, action="store_true")
    parser.add_argument("--hathitrust_root", dest="hathitrust_root", help="Path to HathiTrust")
    parser.add_argument("--unicode_scripts", dest="unicode_scripts", default="data/Scripts.txt", help="Unicode Script.txt file")
    args, rest = parser.parse_known_args()
    
    
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

    
    psf = PairtreeStorageFactory()
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            subcollection, ident = re.match(r"^([^\.]+)\.(.*)$", j["htid"]).groups()
            store = psf.get_store(
                store_dir=os.path.join(
                    args.hathitrust_root,
                    subcollection
                )
            )
            try:
                obj = store.get_object(ident, create_if_doesnt_exist=False)
            except:                
                continue
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
            if args.split_by_script:
                prev_script = None
                cur = ""
                scripts = {}
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

                j["content"] = scripts
            else:
                j["content"] = txt
            ofd.write(json.dumps(j) + "\n")
