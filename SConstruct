import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from steamroller import Environment

# workaround needed to fix bug with SCons and the pickle module
#del sys.modules['pickle']
#sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
#import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file.
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("FOLDS", "", 5),
    ("N", "", 3),
    ("RANDOM_SEED", "", 0),
    ("TRAIN_PROPORTION", "", 0.8),
    ("DEV_PROPORTION", "", 0.1),
    ("TEST_PROPORTION", "", 0.1),    
    ("HATHITRUST_ROOT", "", ""),
    ("HATHITRUST_INDEX_FILENAME", "", "hathi_full_20211001.txt.gz"),
    ("MARC_INDEX_FILENAME", "", "full_marc.json.gz"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/${HATHITRUST_INDEX_FILENAME}"),
    ("MARC_INDEX", "", "${HATHITRUST_ROOT}/${MARC_INDEX_FILENAME}"),
    ("UNICODE_SCRIPTS", "", "data/Scripts.txt"),
    ("PER_LANGUAGE", "", 10),
    ("DATA_LAKE_FILE", "", None),
    ("MAX_DOC_LENGTH","", 400),
    ("RANKED", "", 0),
    ("USE_MIN_SUBDOCS", "", 0),
    # ("PRETRAINED", "", "work/ng_model.pk1.gz"),
    ("PRETRAINED", "", "None"),
    ("SPLIT_BY_SCRIPT", "", False),
    ("APPLY_WINDOW_SIZE", "", 200),
    ("PRECOMPUTED_LID", "", None)
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[],    
    BUILDERS={
        "FilterMarc" : Builder(
            action="python scripts/filter_marc.py --output ${TARGETS[0]} --hathitrust_index ${HATHITRUST_INDEX} --marc_index ${MARC_INDEX}"
        ),
        "CollectionToJSON" : Builder(
            action="python scripts/collection_to_json.py --input ${SOURCES[0]} --output ${TARGETS[0]} --label ${LABEL}"
        ),
        "MergeEntries" : Builder(
            action="python scripts/merge_entries.py --inputs ${SOURCES} --output ${TARGETS[0]}"
        ),
        "ExpandEntries" : Builder(
            action="python scripts/expand_entries.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT} ${'--split_by_script' if SPLIT_BY_SCRIPT else ''}"
        ),
        "RandomSplit" : Builder(
            action="python scripts/random_split.py --input ${SOURCES[0]} --outputs ${TARGETS} --proportions ${TRAIN_PROPORTION} ${DEV_PROPORTION} ${TEST_PROPORTION} --random_seed ${RANDOM_SEED}"
        ),
        "GenerateNegativeExamples" : Builder(
            action="python scripts/generate_negative_examples.py --hathitrust_index ${HATHITRUST_INDEX} --marc_index ${MARC_INDEX} --per_language ${PER_LANGUAGE} --output ${TARGETS[0]} --random_seed ${RANDOM_SEED} --hathitrust_root ${HATHITRUST_ROOT}"
        ),
        "CleanChunkExamples" : Builder(
            action="python scripts/clean_chunk_examples.py --input ${SOURCES[0]} --max_doc_length ${MAX_DOC_LENGTH} --output ${TARGETS[0]}"
        ),
        #"TrainTestSplit" : Builder(
        #    action="python scripts/train_test_split.py --input ${SOURCES[0]} --outputs ${TARGETS} --use_min ${USE_MIN_SUBDOCS} --train_ratio ${TRAIN_PROPORTION} --random_seed ${RANDOM_SEED}"
        #),
        "TrainFasttext" : Builder(
            action="python scripts/train_fasttext.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --model ${TARGETS[0]}"
        ),
        "ApplyFasttext" : Builder(
            action="python scripts/apply_fasttext.py --input ${SOURCES[1]} --model ${SOURCES[0]} ${'--window_size ' + str(APPLY_WINDOW_SIZE) if APPLY_WINDOW_SIZE else ''} --output ${TARGETS[0]}"
        ),
        "BuildLidMatrices": Builder(
            action="python scripts/build_doc_matrices.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "TrainNBModel" : Builder(
            action="python scripts/train_NB_model.py --input ${SOURCES[0]} --model ${TARGETS[0]} --scores ${TARGETS[1]}"
        ),
        "TrainNGModel" : Builder(
            action="python scripts/train_NG_model.py --input ${SOURCES} --model ${TARGETS[0]} --scores --ngram ${N}"
        ),
        "TestModel" : Builder(
            action="python scripts/test_model.py --model ${SOURCES[0]} --input ${SOURCES[1]} --scores ${TARGETS[0]} --ngram ${N} --gen_results ${TARGETS[1]} --at_results ${TARGETS[2]}"
        ),
        "GenerateFinalCorpus" : Builder(
            action="python scripts/generate_final_corpus.py --to_annotate ${SOURCES[0]} --score_files ${SOURCES[1:]} --report ${TARGETS[0]} --corpus ${TARGETS[1]}"
        )
    }
)

env.Decider("timestamp-newer")


# if the data lake file is specified in config.py, no need to build it
if env.get("DATA_LAKE_FILE", None):
    data_lake_with_content = env.File(env["DATA_LAKE_FILE"])
else:
    data_lake = env.FilterMarc(
        "work/data_lake.jsonl.gz",
        [],
        REGEXES=[],
        GRID_MEMORY="4G"
    )
    data_lake_with_content = env.ExpandEntries(
        "work/data_lake_with_content.jsonl.gz",
        data_lake,
        GRID_MEMORY="64G"
    )

if env.get("PRECOMPUTED_LID", None):
    labeled = env.File(env["PRECOMPUTED_LID"])
else:
    armeno_turkish = env.CollectionToJSON(
        "work/True_AT_set.jsonl.gz",
        "data/True_AT.tsv.gz",
        LABEL="armeno_turkish"
    )

    negative_examples = env.GenerateNegativeExamples(
        "work/sampled_negative.jsonl.gz",
        [],
        GRID_MEMORY="64GB"
    )

    armeno_turkish_with_content = env.ExpandEntries(
        "work/labeled_with_content.jsonl.gz",
        armeno_turkish,
        GRID_MEMORY="4GB",
        SPLIT_BY_SCRIPT=True
    )

    combined = env.MergeEntries(
        "work/combined.jsonl.gz",
        [armeno_turkish_with_content, negative_examples],
        GRID_MEMORY="4GB"
    )
    combined_cleaned_chunked = env.CleanChunkExamples(
        "work/chunked_combined.json.gz",
        "work/combined.jsonl.gz",
        GRID_MEMORY="4GB"
    )
    
    train, dev, test = env.RandomSplit(
        ["work/train_data.jsonl.gz", "work/dev_data.jsonl.gz", "work/test_data.jsonl.gz"],
        combined_cleaned_chunked,
        GRID_MEMORY="4G"
    )

    model = env.TrainFasttext(
        "work/fasttext_model.bin",
        [train, dev],
        GRID_MEMORY="8G"
    )

    labeled = env.ApplyFasttext(
        "work/labeled_fasttext.jsonl.gz",
        [model, data_lake_with_content],
        GRID_MEMORY="8G"
    )
    
    lid_matrices = env.BuildLidMatrices(
        ["work/doc_matrices.jsonl.gz"],
        [labeled],
        GRID_MEMORY="8G"
    )

# model, scores = env.TrainNBModel(
#     ["work/nb_model.pk1.gz", "work/nb_scores.json"],
#     ["work/chunked_combined.json.gz"]
# )

# model, scores = env.TrainNBModel(
#     ["work/nb_model.pk1.gz", "work/nb_scores.json"],
#     [train, test]
# )

#model = env.TrainNGModel(
#    "work/ng_model.pkl.gz",
#    train,
#    GRID_MEMORY="4G"
#)

#scores = env.TestModel(
#    ["work/results/${CHUNK_SIZE}/${N}/scores.json", 
#     "work/results/${CHUNK_SIZE}/${N}/gen_results", 
#     "work/results/${CHUNK_SIZE}/${N}/at_results"],
#    [model, test],
#    CHUNK_SIZE = 800,
#    GRID_MEMORY="4G"
#)

# model, scores = env.TrainNBModel(
#     ["work/nb_model.pk1.gz", "work/nb_scores.json"],
#     ["work/chunked_combined.json.gz"]
# )

# ngram_model, ngram_scores = env.TrainNGModel(
#     ["work/ng_model.pk1.gz", "work/NG_scores.json"],
#     ["work/chunked_combined.json.gz"]
# )

#model_scores_pairs = []
#for fold in range(1, env["FOLDS"] + 1):
#   train, dev, test = env.RandomSplit(
#      ["work/${FOLD}/train.jsonl.gz", "work/${FOLD}/dev.jsonl.gz", "work/${FOLD}/test.jsonl.gz"],
#      [labeled_with_content],
#      FOLD=fold,
#      RANDOM_SEED=fold
#   )
#   for n in [2, 3, 4]:
#       model, scores = env.TrainModel(
#           ["work/${FOLD}/${N}/model.pkl.gz", "work/${FOLD}/${N}/scores.json"],
#           [labeled_with_content, train, dev],
#           N=n,
#           FOLD=fold
#       )
#       model_scores_pairs.append((model, scores))
    

#env.GenerateFinalCorpus(
#    ["work/report.txt", "work/final_corpus.jsonl.gz"],
#    [data_lake_with_content] + model_scores_pairs
#    )
