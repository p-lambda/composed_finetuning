import os
from pathlib import Path
import argparse
from collections import defaultdict
import shutil
import uuid
from tqdm import tqdm
import subprocess
import shlex

from format_pairs import setup_clang
from stitch import compile_code, compare_files


class err():
    no_err = 0
    compile_err = 1
    runtime_err = 2
    mismatch_err = 3


def run_code(inp, code):
    # assume no errors in this code
    unique_id = str(uuid.uuid4())[:8]
    compile_errors = compile_code(code, 'prog', unique_id)
    if compile_errors is not None:
        return False, err.compile_err
    objfile = f'prog-{unique_id}'
    inpfile = f'inp-{unique_id}'
    with open(inpfile, 'w') as f:
        for d in inp:
            f.write(d + '\n')
    cmd = f"timeout 2 ./{objfile} < {inpfile}"

    try:
        pred_process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except Exception:
        return False, err.runtime_err

    if pred_process.returncode != 0:
        return False, err.runtime_err
    return True, pred_process.stdout


def oracle_code_check(inp, code, gold_output):
    no_error, output = run_code(inp, code)
    if not no_error:
        if output == err.compile_err:
            return 'comp_err'
        elif output == err.runtime_err:
            return 'runtime_err'
        else:
            raise ValueError("Uncaught")
    else:
        print(f"PRED: {output}")
        print(f"GOLD: {gold_output}")
        if output != gold_output:
            return 'mismatch_err'
        else:
            return 'no_err'


def concat_code(curr_df):
    curr_code = ''
    for i in range(len(curr_df)):
        curr_code += curr_df.iloc[i].code + ' '
    return curr_code.strip()


def parse_preds(path):
    preds = []
    pred_idxs = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('H-'):
                idx, prob, pred = line.split('\t')
                idx = int(idx[2:])
                pred_idxs.append(idx)
                pred = pred.strip()
                # XXX remove UNK
                pred = pred.replace('<unk>', 'unk')
                preds.append(pred)

    # unshuffle
    new_preds = [None] * len(preds)
    for i, pred in zip(pred_idxs, preds):
        new_preds[i] = pred
    preds = new_preds

    # write this to a file
    pred_path = Path(path)
    cleaned_pred_file = pred_path.parent / f'{pred_path.stem}_cleaned.txt'
    with open(cleaned_pred_file, 'w') as f:
        for line in preds:
            f.write(line.strip() + '\n')
    return preds


def evaluate_synthetic(pred, inputs, gold, gold_outputs, parse_only=False, cleaned=False):

    setup_clang('lib')
    code_header = "#include <bits/stdc++.h>\n#include <string>\nusing namespace std;\n\n"

    gold_output_file = Path(gold_outputs).resolve()
    gold_file = Path(gold).resolve()
    pred_path = Path(pred).resolve()

    inputs_ls = []
    with open(Path(inputs).resolve(), 'r') as f:
        for line in f:
            inputs_ls.append(line.strip().split('\t'))

    # work in a tmp folder
    tmp_path = Path(f'tmp_{uuid.uuid4()}').resolve().expanduser()
    tmp_path.mkdir(exist_ok=True)
    os.chdir(str(tmp_path))

    if not gold_output_file.exists():
        golds = []
        with open(gold_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace('#', ' ')
                line = line.replace('$', ' ')
                line = line.replace('~', '\t')
                golds.append(line.strip())

        # preprocess the gold to get outputs
        with open(gold_output_file, 'w') as f:
            for inp, gold in tqdm(zip(inputs_ls, golds)):
                gold = code_header + gold
                no_error, output = run_code(inp, gold)
                if not no_error:
                    raise ValueError(f"gold had an error: {gold}")
                f.write(output + '\n')

    if pred and not cleaned:
        preds = parse_preds(pred_path)

    if cleaned:
        preds = []
        with open(pred_path, 'r') as f:
            for line in f:
                preds.append(line.strip())

    stats = None
    if pred and not parse_only:
        gold_outputs = []
        with open(gold_output_file, 'r') as f:
            for line in f:
                gold_outputs.append(line.strip())

        total = 0
        stats = defaultdict(int)
        for inp, pred, gold_output in tqdm(zip(inputs_ls, preds, gold_outputs)):
            pred = pred.replace('#', ' ')
            pred = pred.replace('$', ' ')
            pred = pred.replace('~', '\t')
            pred = code_header + pred
            result = oracle_code_check(inp, pred, gold_output)

            total += 1
            stats[result] += 1
        stats = {k: f"{v / total: .3f}" for k, v in stats.items()}
        print("RESULTS")
        print(stats)
    os.chdir(str(Path(__file__).resolve().expanduser().parent))
    shutil.rmtree(str(tmp_path))
    return stats
