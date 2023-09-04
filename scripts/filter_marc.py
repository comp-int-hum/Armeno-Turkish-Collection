import json
import gzip
import re
import argparse
import csv

csv.field_size_limit(1000000000)

fields = ["htid", "access", "rights", "ht_bib_key", "description", "source", "source_bib_num", "oclc_num", "isbn", "issn", "lccn", "title", "imprint", "rights_reason_code", "rights_timestamp", "us_gov_doc_flag", "rights_date_used", "pub_place", "lang", "bib_fmt", "collection_code", "content_provider_code", "responsible_entity_code", "digitization_agent_code", "access_profile_code", "author"]

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--hathitrust_index", dest="hathitrust_index", help="Path to HathiTrust index")
    parser.add_argument("--marc_index", dest="marc_index", help="Path to MARC index")
    args, rest = parser.parse_known_args()
    
    is_subset_rx=r"(^|\s)((Armen.*)|(Turkish)|(Ottoman)|(Istʻanpōlta)|(Istʻanpōl)|(Stʻanpōlta)|(Stʻanpōl)|(Beçte)|(Vēnētik)|(Pasmakhanē)|(Mkhitʻarean)|(Constantinople)|(Viēna))" 
    records = {}

    
    def get_id(obj):
        data = json.loads(obj)
        for item in data["fields"]:
            if "974" in item.keys():
                for subf in item["974"]["subfields"]:
                    var_keys = subf.keys()
                    if 'u' in var_keys:
                        return (subf['u'])

    with gzip.open(args.marc_index, 'rt') as ifd:
        for line in ifd:
            j = json.loads(line)
            htid, author, title, author_date, lang, pub_place, pub_date = get_info(j)            
            if re.search(is_subset_rx, line, re.IGNORECASE) or lang in ["ota", "tur"]: 
                records[htid] = {"htid" : htid, "marc_meta" : j}

    with gzip.open(args.hathitrust_index, "rt") as ifd, gzip.open(args.output, 'wt') as ofd:
        c_ifd = csv.DictReader(
            ifd,
            fieldnames=fields,
            delimiter="\t"
        )

        for row in c_ifd:
            if row["htid"] in records:
                records[row["htid"]]["hathi_meta"] = row
                ofd.write(json.dumps(records[row["htid"]]) + "\n")
