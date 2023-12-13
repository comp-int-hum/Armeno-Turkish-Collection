import json
import gzip
import logging
import argparse
import numpy as np
logger = logging.getLogger("doc_id")

class CustomNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # label_list = ['gmh_Latin', 'pol_Latin', 'zxx_Latin', 'heb_Hebrew', 'tur_Arabic', 'sla_Cyrillic', 'fro_Latin', 'rum_Cyrillic', 'dan_Latin', 'kan_Latin', 'srp_Cyrillic', 'yid_Hebrew', 'rum_Greek', 'fin_Latin', 'mul_Latin', 'chu_Han', 'hun_Latin', 'gle_Latin', 'per_Arabic', 'enm_Latin', 'san_Devanagari', 'ara_Arabic', 'haw_Greek', 'iri_Latin', 'tur_Armenian', 'tha_Thai', 'ind_Thai', 'chu_Cyrillic', 'hrv_Cyrillic', 'kor_Hangul', 'chi_Latin', 'ota_Armenian', 'haw_Latin', 'chi_Han', 'cat_Latin', 'por_Latin', 'lav_Greek', 'hin_Latin', 'tam_Tamil', 'epo_Latin', 'pli_Latin', 'fry_Latin', 'dut_Latin', 'eng_Latin', 'hrv_Latin', 'syr_Arabic', 'roh_Latin', 'ota_Arabic', 'gem_Latin', 'ice_Latin', 'ind_Latin', 'hin_Arabic', 'bre_Latin', 'jav_Latin', 'slv_Latin', 'jav_Lao', 'rus_Cyrillic', 'spa_Latin', 'jpn_Han', 'ang_Latin', 'may_Han', 'hin_Devanagari', 'jav_Thai', 'und_Latin', 'kor_Han', 'ita_Latin', 'tur_Hebrew', 'map_Latin', 'syr_Latin', 'jav_Arabic', 'gre_Greek', 'scc_Cyrillic', 'ukr_Cyrillic', 'san_Latin', 'grc_Latin', 'tgl_Latin', 'may_Latin', 'chu_Latin', 'gla_Latin', 'kan_Kannada', 'fre_Latin', 'jpn_Katakana', 'syr_Syriac', 'frm_Latin', 'lat_Latin', 'mar_Devanagari', 'lit_Latin', 'urd_Arabic', 'est_Cyrillic', 'arm_Latin', 'srp_Latin', 'swe_Latin', 'lav_Latin', 'slo_Latin', 'may_Arabic', 'pli_Thai', 'nor_Latin', 'roa_Latin', 'scr_Cyrillic', 'tam_Latin', 'est_Latin', 'pro_Latin', 'bul_Cyrillic', 'cze_Latin', 'sla_Latin', 'jpn_Hiragana', 'wel_Latin', 'est_Arabic', 'jpn_Arabic', 'afr_Latin', 'ger_Latin', 'grc_Greek', 'scr_Latin', 'arm_Armenian', 'gre_Latin', 'rum_Latin', 'oci_Latin', 'fri_Latin', 'heb_Latin', 'bul_Latin', 'mul_Greek', 'per_Latin', 'haw_Cyrillic','epo_Arabic', 'und_Cyrillic', 'hun_Greek','pli_Sinhala','lit_Cyrillic','ind_Lao','jpn_Latin','spa_Katakana','zxx_Cyrillic','zxx_null','syr_null','sla_null','wel_null','mul_null','tur_Latin']
    # lang2id = {l : i for i, l in enumerate(label_list)}
    results = []
    AT_results = []
    documents = []
    with gzip.open("work/labeled_fasttext.jsonl.gz", "rt") as ifd, gzip.open("work/doc_matrices.jsonl.gz", "wt") as ofd: # "rb" might be preferred here; though it might not matter much
        for line in ifd:
            entry = json.loads(line)
            ini_dict_list = entry["lid"]
            htid = entry["htid"]
            label_list = list({key for d in ini_dict_list for key in d.keys()}) # create list of all keys
            lang2id = {l : i for i, l in enumerate(label_list)}
            column_len = len(lang2id) # number of languages
            row_len = len(ini_dict_list) # number of windows / "subdocs"
            if not column_len or not row_len:
                logger.info(f"Document with htid {htid} has label_size {column_len} and subdoc length {row_len}")
                continue

            # Initialize doc_matrix to zeros
            doc_matrix = np.zeros((column_len, row_len)) # Shape: lang x subdoc_num

            # Iterate through subdocs and build doc_matrix
            for subdoc_num, scores in enumerate(ini_dict_list): # iterate through list of lid dicts
                for lang, score in scores.items():
                    doc_matrix[lang2id[lang], subdoc_num] = score

            # Clean final doc matrix and convert to list
            cleaned_doc_matrix = np.nan_to_num(doc_matrix)
            full_doc_matrix = cleaned_doc_matrix

            # Create max_doc_matrix
            total_probs = np.sum(cleaned_doc_matrix, axis=1) # sum across all subdocs to get total prob per lang
            max_lang_index = np.argmax(total_probs) # get the index of the lang with highest total prob
            max_lang = label_list[max_lang_index] # get the lang name of max_lang_index

            # Create binary max list
            max_lang_indices = np.argmax(cleaned_doc_matrix, axis=0) # get index of max lang at each timestep
            bool_max_matrix = max_lang_indices == max_lang_index # creates 1d matrix of booleans

            # Write output
            doc = {
                "htid" : entry["htid"],
                "full_doc_matrix" : full_doc_matrix,
                "max_doc_matrix": full_doc_matrix[max_lang_index],
                "label_list": label_list,
                "max_lang_index": max_lang_index,
                "max_lang": max_lang,
                "binary_max": bool_max_matrix.astype(int)
            }
            documents.append(doc) # for future use
            ofd.write(json.dumps(doc, cls=CustomNumpyEncoder) + "\n")
