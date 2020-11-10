import os
from pathlib import Path
import argparse
import pandas as pd
from typing import NamedTuple
from collections import defaultdict
import shutil
import uuid
from tqdm import tqdm
import numpy as np

from format_pairs import setup_clang
from stitch import compile_code, run_tests


class err():
    no_err = 0
    compile_err = 1
    runtime_err = 2
    mismatch_err = 3


def oracle_code_check(code, probid, subid):
    unique_id = probid + "-" + subid
    # generate c++
    compile_errors = compile_code(code, probid, subid)
    if compile_errors is not None:
        # cleanup(unique_id)
        return 'comp_err'
    # run testcases
    test_errors, _ = run_tests(code, probid, subid, 'testcases')
    if test_errors == err.no_err:
        return 'no_err'
    elif test_errors == err.mismatch_err:
        return 'mismatch_err'
    elif test_errors == err.runtime_err:
        return 'runtime_err'
    else:
        return 'testcase_err'


def concat_code(curr_df):
    curr_code = ''
    for i in range(len(curr_df)):
        curr_code += curr_df.iloc[i].code + ' '
    return curr_code.strip()


def parse_preds(path):
    preds = []
    idxs = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('H-'):
                continue
            toks = line.split('\t')
            idx = toks[0]
            prob = toks[1]
            pred = '\t'.join(toks[2:])
            idx = int(idx[2:])
            idxs.append(idx)
            preds.append(pred.strip())
    # unshuffle
    new_preds = [None] * len(preds)
    for i, pred in zip(idxs, preds):
        new_preds[i] = pred
    preds = new_preds

    # write this to a file
    pred_path = Path(path)
    cleaned_pred_file = pred_path.parent / f'{pred_path.stem}_cleaned.txt'
    with open(cleaned_pred_file, 'w') as f:
        for line in preds:
            f.write(line.strip() + '\n')

    return preds


def parse_orig_preds(path):
    preds = []
    idxs = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('S-'):
                continue
            toks = line.split('\t')
            idx = toks[0]
            pred = '\t'.join(toks[1:])
            idx = int(idx[2:])
            idxs.append(idx)
            preds.append(pred.strip())
    # unshuffle
    new_preds = [None] * len(preds)
    for i, pred in zip(idxs, preds):
        new_preds[i] = pred
    preds = new_preds
    return preds


def parse_classifier_preds(path):
    preds = []
    idxs = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('H-'):
                continue
            toks = line.split('\t')
            idx = toks[0]
            prob = float(toks[1])
            pred = int(toks[2])
            idx = int(idx[2:])
            idxs.append(idx)

            # threshold
            prob = 1.0 / (1.0 + np.exp(-prob))
            pred = int(prob > 0.9)
            preds.append(pred)
    # unshuffle
    new_preds = [None] * len(preds)
    for i, pred in zip(idxs, preds):
        new_preds[i] = pred
    preds = new_preds
    return preds


def evaluate_full_spoc(id_path, pred, pred_cleaned=False, parse_only=False):
    pred = str(pred)

    id_path = Path(id_path).resolve().expanduser()

    if not pred_cleaned:
        preds = parse_preds(pred)
    else:
        preds = []
        with open(pred, 'r') as f:
            for line in f:
                preds.append(line.strip())

    # if exists, parse classifier file
    filter_by_cls = (pred is not None) and (('_pi' in pred) or ('denoise' in pred))
    if filter_by_cls:
        cls_pred_path = Path(pred)
        cls_pred_path = cls_pred_path.parent / f'{cls_pred_path.stem}_classifier.txt'
        if not cls_pred_path.exists():
            filter_by_cls = False
        else:
            cls_preds = parse_classifier_preds(cls_pred_path)
            orig_preds = parse_orig_preds(pred)
    print("FILTER BY CLS:", filter_by_cls)

    if not parse_only:
        setup_clang('lib')

        # work in a tmp folder
        tmp_path = Path(f'tmp_{uuid.uuid4()}').resolve().expanduser()
        tmp_path.mkdir(exist_ok=True)
        os.chdir(str(tmp_path))

        ids = []
        with open(id_path, 'r') as f:
            for line in f:
                ids.append(line.strip())

        code_header = "#include <bits/stdc++.h>\n#include <string>\nusing namespace std;\n\n"
        total = 0
        stats = defaultdict(int)
        for i, (id, pred) in tqdm(enumerate(zip(ids, preds))):
            probid, subid, workerid = id.split('-')
            if filter_by_cls and cls_preds[i] == 0:
                # Replace only if classifier thinks its wrong
                pred = orig_preds[i]

            pred = code_header + pred
            pred = pred.replace('$', ' ')
            pred = pred.replace('~', '\t')
            pred = pred.replace('<unk>', 'unk')
            result = oracle_code_check(pred, probid, subid)

            total += 1
            stats[result] += 1
        stats = {k: f"{v / total: .3f}" for k, v in stats.items()}
        print("RESULTS")
        print(stats)
        os.chdir(str(Path(__file__).resolve().expanduser().parent))
        shutil.rmtree(str(tmp_path))
        return stats
