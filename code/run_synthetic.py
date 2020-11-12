from pathlib import Path
import numpy as np
import sentencepiece as spm
import shutil
import shlex
import subprocess
from copy import deepcopy
from collections import defaultdict
import argparse
import pandas as pd

from c_tokenizer_mod import C_Tokenizer
from evaluate_synthetic import evaluate_synthetic


GLOBAL_SUFFIX = ""
NUM_NAMES = 1000
USER_DIR = 'fairseq_modules/'
ROOT = '.'


def strip_ext(path):
    return path.parent / path.stem


class Synthetic:

    def __init__(self, root, split='train', denoise=False, num_examples=1000, num_unlabeled=20000):
        self.split = split
        self.root = Path(root)
        self.id = []
        if denoise:
            self.id.append('denoise')
        if num_unlabeled != 20000:
            self.id.append(f'numunlabeled{num_unlabeled}')

        self.id.append(str(num_examples))
        self.id = '_'.join(self.id)
        self.id += GLOBAL_SUFFIX

        self.data_dir = self.root / f'synthetic_{self.id}'
        self.denoise = denoise
        # alias
        self.vocab_size = 600
        self.spm_model_prefix = self.data_dir / 'sentencepiece.bpe'

        self.tokenizer = C_Tokenizer()

        self.types = ['int', 'bool', 'string']
        self.two_arg_funcs = {'int': ['max', 'min']}

        self.data_dir.mkdir(exist_ok=True)
        if denoise:
            if self.split == 'train':
                self.create_examples(num_unlabeled)
            else:
                self.create_examples(500)
        else:
            if self.split == 'train':
                self.create_examples(num_examples)
            else:
                self.create_examples(500)

    def get_string_with_prefix(self, prefix, num=1):
        ints = np.random.choice(list(range(10)), size=num, replace=False)
        outputs = [f'{prefix}_{i}' for i in ints]
        if num == 1:
            return outputs[0]
        else:
            return outputs

    def get_var(self, num=1):
        return self.get_string_with_prefix('var', num=num)

    def get_string(self, num=1):
        if num == 1:
            return '\"' + self.get_string_with_prefix('str', num=num) + '\"'
        else:
            return [f'"{s}"' for s in self.get_string_with_prefix('str', num=num)]

    def get_value(self, type_str):
        if type_str == 'int':
            return np.random.randint(0, 100)
        elif type_str == 'bool':
            if (np.random.rand() > 0.5):
                return 'true'
            else:
                return 'false'
        elif type_str == 'string':
            return self.get_string()

    def get_print_subtype(self, out_str):
        code = f"cout << {out_str}; "
        pseudocode = np.random.choice([
                f"print {out_str}; ",
                f"output {out_str} to stdout; "])

        noised_code = ""

        return "", pseudocode, code, noised_code

    def get_input_subtype(self, var_list, type_list):
        var_list_str = ' >> '.join(var_list)
        code = f"cin >> {var_list_str}; "

        listed_var_list_str = ', '.join(var_list)
        noised_code = np.random.choice([
            f"cin {var_list_str}; ",
            f"cin >> {listed_var_list_str}; ",
            f"cin << {var_list_str}; "])

        pseudocode = f"read {listed_var_list_str} from stdin; "

        inputs = []
        for type_str in type_list:
            if type_str == 'bool':
                inputs.append(str(np.random.randint(2)))
            else:
                inputs.append(str(self.get_value(type_str)))
        inputs = '\t'.join(inputs)
        return inputs, pseudocode, code, noised_code

    def get_initialize_subtype(self, type_str, var_str, value, set_only=False):
        if not set_only:
            code = np.random.choice([
                f"{type_str} {var_str} = {value}; ",
                f"{type_str} {var_str}; {var_str} = {value}; "])
            other_types = list(set(self.types) - set([type_str]))
            noised_code = np.random.choice([
                        f"{np.random.choice(other_types)} {var_str} = {value}; ",
                        f"{var_str} = {value}; ",
                        f"{np.random.choice(other_types)} {var_str}; {var_str} = {value}; ",
                        # f"{type_str} {var_str}; {self.get_var()} = {value}; ",
                        ])
        else:
            code = f"{var_str} = {value}; "
            noised_code = np.random.choice([
                f"{type_str} {var_str} = {value}; ",
                f"{np.random.choice(self.types)} {var_str} = {value}; "])
        # ambiguous pseudocode
        pseudocode = f"set {var_str} to {value}; "

        return "", pseudocode, code, noised_code

    def get_instantiate_subtype(self, var_list, type_str):
        code = f"{type_str} {', '.join(var_list)}; "
        pseudocode = f"instantiate {', '.join(var_list)}; "
        other_types = list(set(self.types) - set([type_str]))
        noised_code = f"{np.random.choice(other_types)} {', '.join(var_list)}; "
        return "", pseudocode, code, noised_code

    def get_type_preserving_subtype(self, var_str, type_str, curr_vars):
        # ambiguity in add and set
        if type_str == 'int':
            value = self.get_value(type_str)
            type_idx = np.random.randint(3)
            if type_idx == 0:
                code = np.random.choice([f'{var_str} += {value}; ', f'{var_str} = {var_str} + {value}; '])
                pseudocode = np.random.choice([
                        f'set {var_str} to {value} plus {var_str}; '])
                noised_code = np.random.choice([
                            f'{type_str} {var_str} += {value}; ',
                            f'{type_str} {var_str} = {var_str} + {value}; ',
                            f'{np.random.choice(self.types)} {var_str} += {value}; ',
                            f'{np.random.choice(self.types)} {var_str} = {var_str} + {value}; '])
            elif type_idx == 1:
                code = np.random.choice([f'{var_str} -= {value}; ', f'{var_str} = {var_str} - {value}; '])
                pseudocode = np.random.choice([
                        f'set {var_str} to {value} minus {var_str}; '])
                other_var = self.get_var()
                noised_code = np.random.choice([
                            f'{type_str} {var_str} -= {value}; ',
                            f'{type_str} {var_str} = {var_str} - {value}; ',
                            f'{np.random.choice(self.types)} {var_str} -= {value}; ',
                            f'{np.random.choice(self.types)} {var_str} = {var_str} - {value}; '])
            elif type_idx == 2:
                _, pseudocode, code, noised_code = self.get_initialize_subtype("", var_str, value, set_only=True)
        elif type_str == 'bool':
            value = self.get_value(type_str)
            type_idx = np.random.randint(3)
            if type_idx == 0:
                pseudocode = f'set {var_str} to its negation; '
                code = f'{var_str} = !{var_str}; '
                noised_code = np.random.choice([
                   f'{type_str} {var_str} = !{var_str}; ',
                   f'{np.random.choice(self.types)} {var_str} = !{var_str}; '])
            if type_idx == 0:
                code = np.random.choice([
                    f'{var_str} = {var_str} && {value}; '])
                pseudocode = f'set {var_str} to the and with {value}; '
                noised_code = np.random.choice([
                    f'{type_str} {var_str} = {var_str} && {value}; ',
                    f'{np.random.choice(self.types)} {var_str} = {var_str} && {value}; ',])
            elif type_idx == 1:
                _, pseudocode, code, noised_code = self.get_initialize_subtype("", var_str, value, set_only=True)
            elif type_idx == 2:
                # execute a variable swap. test understanding of scope and relation to initialization
                type_list = list(set(list(zip(*curr_vars.values()))[0]))
                swap_type = None
                for curr_swap_type in type_list:
                    vars_with_type = [k for k in curr_vars.keys()
                                      if curr_vars[k][0] == curr_swap_type]
                    if len(vars_with_type) < 2:
                        continue
                    swap_type = curr_swap_type
                    var_1, var_2 = vars_with_type[0], vars_with_type[1]
                    break

                code = f"if ({var_str})" + " { "
                pseudocode = f"if {var_str} is true, "
                noised_code = code
                if swap_type is None:
                    swap_type = np.random.choice(self.types)
                    var_1, var_2 = self.get_var(num=2)
                    val_1, val_2 = self.get_value(swap_type), self.get_value(swap_type)
                    _, pc_1, c_1, n_1 = self.get_initialize_subtype(swap_type, var_1, val_1)
                    _, pc_2, c_2, n_2 = self.get_initialize_subtype(swap_type, var_2, val_2)
                    code += '~ ' + c_1 + '~ ' + c_2
                    pseudocode += 'TAB ' + pc_1 + 'TAB ' +  pc_2
                    noised_code += '~ ' + n_1 + '~ ' + n_2

                other_types = list(set(self.types) - set([swap_type]))
                noised_code += np.random.choice([
                        f"temp = {var_1}; {var_1} = {var_2}; {var_2} = temp; {'}'} ",
                        f"{np.random.choice(other_types)} temp = {var_1}; {var_1} = {var_2}; {var_2} = temp; {'}'} ",])

                code += f'{swap_type} temp = {var_1}; '
                code += f'{var_1} = {var_2}; '
                code += f'{var_2} = temp;' + ' } '
                pseudocode += np.random.choice([
                    f'set {var_1} to the value of {var_2} and {var_2} to the value of {var_1}; ',
                    f'swap the values of {var_1} and {var_2}; '])

        elif type_str == 'string':
            value = self.get_string()
            type_idx = np.random.randint(3)
            if type_idx == 0:
                code = np.random.choice([
                    f'{var_str} += {value}; ',
                    f'{var_str} = {var_str} + {value}; '])
                pseudocode = f'add {value} to the end of {var_str}; '
                noised_code = np.random.choice([
                        f'{np.random.choice(self.types)} {var_str} = {var_str} + {value}; ',
                        f"{np.random.choice(self.types)} {var_str} += {value}; ",
                        f'{type_str} {var_str} = {var_str} + {value}; ',
                        f"{type_str} {var_str} += {value}; "])
            elif type_idx == 1:
                code = f'{var_str} = {value} + {var_str}; '
                pseudocode = f'add {value} to the beginning of {var_str}; '
                noised_code = np.random.choice([
                        f'{np.random.choice(self.types)} {var_str} = {value} + {var_str}; ',
                        f'{type_str} {var_str} = {value} + {var_str}; '])
            elif type_idx == 2:
                _, pseudocode, code, noised_code = self.get_initialize_subtype(
                        "", var_str, value, set_only=True)
        else:
            raise ValueError(f"{type_str} not found")
        return "", pseudocode, code, noised_code

    def get_func_subtype(self, func, var_str1, var_str2, var_set):
        return (
            "",
            f"set {var_set} to {func} applied to {var_str1} and {var_str2}; ",
            f"{var_set} = {func}({var_str1}, {var_str2}); ",
            f"{np.random.choice(self.types)} {var_set} = {func}({var_str1}, {var_str2}); ")

    def create_example_type(self, type_idx):
        header = "int main () { "
        footer = " return 0; } "

        num_var_ub = 10
        num_var_limit = 4

        num_vars = np.random.randint(1, num_var_limit)
        var_strs = self.get_var(num=num_vars)
        if num_vars == 1:
            var_strs = [var_strs]
        vars_by_type = defaultdict(list)
        curr_vars = {}
        for i, var_str in zip(range(num_vars), var_strs):
            type_str = np.random.choice(self.types)
            value = self.get_value(type_str)
            curr_vars[var_str] = (type_str, value)
            vars_by_type[type_str].append(var_str)

        outs = []
        for var_str, (type_str, value) in curr_vars.items():
            if np.random.rand() < 0.5:
                outs.append(self.get_initialize_subtype(type_str, var_str, value))
            else:
                outs += [self.get_instantiate_subtype([var_str], type_str),
                         self.get_input_subtype([var_str], [type_str])]

        # make sure all of them are touched to resolve ambiguity
        varlist = list(curr_vars.keys())
        np.random.shuffle(varlist)
        for i in range(len(curr_vars)):
            varname = varlist[i]
            outs.append(self.get_type_preserving_subtype(varname, curr_vars[varname][0], curr_vars))

        # random mix of setting the variables, adding, concatting, negating, or initializing new vars
        num_line_limit = 5

        for i in range(np.random.randint(1, num_line_limit)):
            num_vars = len(curr_vars)
            if np.random.rand() < 0.8:
                if np.random.rand() < 0.1:
                    # sometimes add a func
                    if len(vars_by_type['int']) > 1:
                        func = np.random.choice(self.two_arg_funcs['int'])
                        var1, var2 = np.random.choice(vars_by_type['int'], size=2, replace=False)
                        outs.append(self.get_func_subtype(func, var1, var2, var1))

                # pick a random variable in curr_vars
                varname = list(curr_vars.keys())[np.random.randint(num_vars)]
                outs.append(self.get_type_preserving_subtype(varname, curr_vars[varname][0], curr_vars))
            else:
                # max vars is 10
                if len(curr_vars) >= num_var_ub:
                    continue
                type_str = np.random.choice(self.types)
                different = False
                while not different:
                    var_str = self.get_var()
                    different = (var_str not in curr_vars)

                value = self.get_value(type_str)
                curr_vars[var_str] = (type_str, value)
                outs.append(self.get_initialize_subtype(type_str, var_str, value))

        # print all
        for var_str in curr_vars.keys():
            outs.append(self.get_print_subtype(var_str))

        inputs, pseudocode, code, noised_code = zip(*outs)

        if self.denoise:
            # with some probability, keep it the same
            noised_code_ret = list(deepcopy(code))
            if np.random.rand() < 0.9:
                # pick one to three noised versions and replace
                for j in np.random.randint(len(code), size=np.random.randint(1, 4)):
                    noised_code_ret[j] = noised_code[j]

            # tokenize first
            noised_code_ret = [' '.join(self.tokenizer.tokenize(c)[0]) for c in noised_code_ret]
            code = [' '.join(self.tokenizer.tokenize(c)[0]) for c in code]

            sep = ' $ '
            noised_code_ret = sep.join(noised_code_ret)
            noised_code_ret = header.strip() + sep + noised_code_ret + sep + footer.strip()
            code = sep.join(code)
            return "", noised_code_ret, header.strip() + sep + code + sep + footer.strip()
        else:
            code = [' '.join(self.tokenizer.tokenize(c)[0]) for c in code]

            sep = ' $ '
            pseudocode = sep + sep.join(pseudocode) + sep
            code = sep.join(code)
            inputs = '\t'.join([s.strip('\"') for s in inputs if len(s) > 0])
            return inputs, pseudocode, header.strip() + sep + code + sep + footer.strip()


    def create_examples(self, n):
        self.inp_path = self.data_dir / f'{self.split}.inp'
        self.src_path = self.data_dir / f'{self.split}.src'
        self.tgt_path = self.data_dir / f'{self.split}.tgt'

        if not self.src_path.exists() or not self.tgt_path.exists():
            pseudocodes = []
            codes = []
            inputs = []
            for i in range(n):
                type_idx = np.random.randint(0, 3)
                inp, pc, c = self.create_example_type(type_idx)
                if not self.denoise:
                    inputs.append(inp)
                pseudocodes.append(pc)
                codes.append(c)

            with open(self.src_path, 'w') as f:
                for line in pseudocodes:
                    f.write(line + '\n')

            with open(self.tgt_path, 'w') as f:
                for line in codes:
                    f.write(line + '\n')

            if not self.denoise:
                with open(self.inp_path, 'w') as f:
                    for line in inputs:
                        f.write(line + '\n')

    def train_spm(self, paths):
        # def train_spm(self):
        paths = [str(p) for p in paths]
        if not Path(f"{str(self.spm_model_prefix)}.model").exists() or not Path(f"{str(self.spm_model_prefix)}.vocab").exists():
            path_list = ','.join(paths)
            cmd = f'--input={path_list} \
                    --model_prefix={self.spm_model_prefix} \
                    --vocab_size={self.vocab_size} \
                    --character_coverage=1.0 \
                    --model_type=bpe'
            spm.SentencePieceTrainer.Train(cmd)


def parse_pred_file(pred_file):
    pred_file = Path(pred_file)
    preds = []
    idxs = []
    with open(pred_file, 'r') as f:
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

    # clean the std outputs
    pred_file_clean = pred_file.parent / f'{pred_file.stem}_cleaned.txt'
    with open(pred_file_clean, 'w') as f:
        for line in preds:
            f.write(line.strip() + '\n')
    return pred_file_clean


# train and encode spm
def encode(in_paths, spm_model_prefix):
    out_paths = [str(p.parent / f'{p.stem}.bpe{p.suffix}') for p in in_paths]
    in_paths = [str(p) for p in in_paths]
    if not Path(out_paths[0]).exists() or not Path(out_paths[1]).exists():
        cmd = f"python spm_encode.py \
                --model {spm_model_prefix}.model \
                --output_format=piece \
                --inputs {' '.join(in_paths)} \
                --outputs {' '.join(out_paths)}"
        subprocess.run(shlex.split(cmd))
    out_paths = [Path(p) for p in out_paths]
    return out_paths



def direct(num_examples=1000, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'direct_synthetic_{ds_train.id}_dropout{dropout}'

    # train sentence piece with train data
    paths = [ds_train.src_path, ds_train.tgt_path]
    ds_train.train_spm(paths)
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    src = 'src'
    tgt = 'tgt'
    model = 'transformer'
    criterion = 'cross_entropy'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{exp_id}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-train \
            {ds_train.data_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers 3 --decoder-layers 3 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 2 --decoder-attention-heads 2 \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 400 --warmup-init-lr 5e-4 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 500 --save-interval 500 --save-dir {save_dir} --user-dir {USER_DIR}"
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate \
            {ds_test.data_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {save_dir}/checkpoint_best.pt \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path, inputs=ds_test.inp_path,
                                    gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id
    stats.append(res)


def denoise(small=False, num_examples=1000, do_eval=True, num_unlabeled=20000, dropout=0.1):
    root = Path(ROOT)
    std_ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    std_ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    std_ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)
    ds_train = Synthetic(root / 'data', split='train', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled)
    ds_val = Synthetic(root / 'data', split='val', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled)
    ds_test = Synthetic(root / 'data', split='test', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled)

    exp_id = f'denoise_synthetic_{std_ds_train.id}_numunlabeled{num_unlabeled}'
    if small:
        exp_id += '_small'
    std_exp_id = f'direct_synthetic_{std_ds_train.id}_dropout{dropout}'

    paths = [ds_train.src_path, ds_train.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    src = 'src'
    tgt = 'tgt'
    model = 'transformer'
    criterion = 'cross_entropy'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{exp_id}.txt'

    srcdict = std_ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    if small:
        params = "--encoder-layers 3 --decoder-layers 3 \
               --encoder-embed-dim 256 --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 2 --decoder-attention-heads 2 "
    else:
        params = "--encoder-layers 5 --decoder-layers 5 \
               --encoder-embed-dim 256 --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 4 --decoder-attention-heads 4 "

    cmd = f"fairseq-train \
            {ds_train.data_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
           --lr 1e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 50 --save-interval 50 --save-dir {save_dir} --user-dir {USER_DIR}"
    subprocess.run(shlex.split(cmd))

    if do_eval:
        std_cleaned_pred_path = root / f'results/{std_exp_id}_cleaned.txt'
        eval_dir = ds_train.data_dir / 'evaluate_2step'
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(exist_ok=True)
        eval_src_path = eval_dir / 'testp.src'
        eval_tgt_path = eval_dir / 'testp.tgt'
        if not eval_src_path.exists():
            shutil.copy(std_cleaned_pred_path, eval_src_path)
        if not eval_tgt_path.exists():
            shutil.copy(std_ds_test.tgt_path, eval_tgt_path)

        # sentencepiece
        paths = [eval_src_path, eval_tgt_path]
        out_paths = encode(paths, std_ds_train.spm_model_prefix)

        cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
                --testpref {strip_ext(out_paths[0])} --destdir {eval_dir} \
                --joined-dictionary --workers 2 --srcdict {srcdict}'
        subprocess.run(shlex.split(cmd))

        # EVALUATE
        cmd = f"fairseq-generate {eval_dir} \
            --source-lang {src} --target-lang {tgt} \
            --gen-subset test \
            --path {save_dir}/checkpoint_best.pt \
            --beam 1 \
            --remove-bpe=sentencepiece --user-dir {USER_DIR}"
        with open(results_path, 'w') as f:
            subprocess.run(shlex.split(cmd), stdout=f)

        res = evaluate_synthetic(pred=results_path, inputs=std_ds_test.inp_path,
                                 gold=std_ds_test.tgt_path, gold_outputs=std_ds_test.data_dir / 'test.out')
        res['exp_id'] = exp_id + " (direct + denoise)"
        stats.append(res)
    return exp_id


def composed(num_examples=1000, pretrain_init=False, eval_only=False, num_unlabeled=20000, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    if pretrain_init:
        exp_id = f'composed_synthetic_pretrain_{ds_train.id}'
    else:
        exp_id = f'composed_synthetic_{ds_train.id}'

    denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'

    suffix = f'_numunlabeled{num_unlabeled}_dropout{dropout}'
    std_suffix = f'_dropout{dropout}'
    exp_id += suffix

    if pretrain_init:
        std_exp_id = f'pretrain_synthetic_{ds_train.id}'
        std_exp_id += suffix
    else:
        std_exp_id = f'direct_synthetic_{ds_train.id}'
        std_exp_id += std_suffix

    src = 'src'
    tgt = 'tgt'
    model = 'double_transformer'
    criterion = 'reinforce_criterion'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{exp_id}.txt'
    results_path_2 = root / f'results/{exp_id}_pi.txt'

    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    standard_save_path = root / f'models/{std_exp_id}/checkpoint_best.pt'

    weight_decay = 0.0001
    ce_loss_lambda = 1.0

    if not eval_only:
        cmd = f"fairseq-train \
                {ds_train.data_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers 3 \
               --decoder-layers 3 \
               --encoder-embed-dim 256 \
               --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 \
               --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 2 \
               --decoder-attention-heads 2 \
               --encoder-layers-2 5 \
               --decoder-layers-2 5 \
               --encoder-embed-dim-2 256 \
               --decoder-embed-dim-2 256 \
               --encoder-ffn-embed-dim-2 1024 \
               --decoder-ffn-embed-dim-2 1024 \
               --encoder-attention-heads-2 4 \
               --decoder-attention-heads-2 4 \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay {weight_decay} \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 90 --warmup-init-lr 1e-7 \
               --lr 2e-4 --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch 50 --save-interval 50 --save-dir {save_dir} --user-dir {USER_DIR} \
               --keep-best-checkpoints 4 \
               --pi-restore-path {denoise_save_path} \
               --f-restore-path {standard_save_path} \
               --ce-loss-lambda {ce_loss_lambda} \
               --best-checkpoint-metric loss_2 \
               --label-smoothing 0.0 --sampling"
        subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate \
            {str(ds_test.data_dir)} \
        --source-lang {src} --target-lang {tgt} \
        --path {save_dir}/checkpoint_best.pt \
        --gen-subset test \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id + " (base predictor)"
    stats.append(res)

    cleaned_pred_path = str(root / f'results/{exp_id}_cleaned.txt')
    eval_dir = ds_train.data_dir / 'evaluate_2step_stacked'
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(exist_ok=False)
    eval_src_path = eval_dir / 'testp.src'
    eval_tgt_path = eval_dir / 'testp.tgt'
    shutil.copy(str(cleaned_pred_path), eval_src_path)
    if not eval_tgt_path.exists():
        shutil.copy(ds_test.tgt_path, eval_tgt_path)

    # sentencepiece
    paths = [eval_src_path, eval_tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)

    srcdict = ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {strip_ext(out_paths[0])} --destdir {str(eval_dir)} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate {eval_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {denoise_save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path_2, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path_2, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id
    stats.append(res)


def pretrained(num_examples=1000, num_unlabeled=20000, dropout=0.1):
    # pretrain a small denoiser first
    denoise_exp_id = denoise(small=True, num_examples=num_examples, num_unlabeled=num_unlabeled)
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'pretrain_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}_dropout{dropout}'

    denoise_large_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'

    paths = [ds_train.src_path, ds_train.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    src = 'src'
    tgt = 'tgt'
    model = 'transformer'
    criterion = 'cross_entropy'
    save_dir = root / f'models/{exp_id}'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    denoise_large_save_path = root / f'models/{denoise_large_exp_id}/checkpoint_best.pt'
    results_path = root / f'results/{exp_id}.txt'
    results_path_2 = root / f'results/{exp_id}_pi.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-train \
            {ds_train.data_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers 3 --decoder-layers 3 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 2 --decoder-attention-heads 2 \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 150 --warmup-init-lr 1e-4 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 300 --save-interval 300 --save-dir {save_dir} --user-dir {USER_DIR} \
           --restore-file {denoise_save_path} \
           --reset-optimizer \
           --reset-lr-scheduler \
           --reset-dataloader \
           --reset-meters"
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate {ds_test.data_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {save_dir}/checkpoint_best.pt \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id
    stats.append(res)

    # EVAL SECOND STEP

    cleaned_pred_path = root / f'results/{exp_id}_cleaned.txt'
    eval_dir = ds_train.data_dir / 'evaluate_2step_pretrain'
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(exist_ok=False)
    eval_src_path = eval_dir / 'testp.src'
    eval_tgt_path = eval_dir / 'testp.tgt'
    shutil.copy(cleaned_pred_path, eval_src_path)
    if not eval_tgt_path.exists():
        shutil.copy(ds_test.tgt_path, eval_tgt_path)

    # sentencepiece
    paths = [eval_src_path, eval_tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)

    srcdict = ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {strip_ext(out_paths[0])} --destdir {str(eval_dir)} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate {eval_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {denoise_large_save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path_2, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path_2, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id + " (pretrain + denoise)"
    stats.append(res)


def backtranslation(num_examples=1000, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)
    denoise_ds_train = Synthetic(root / 'data', split='train', denoise=True, num_examples=num_examples)

    exp_id = f'backtranslation_synthetic_{ds_train.id}_dropout{dropout}'

    # train spm with all data
    paths = [ds_train.src_path, ds_train.tgt_path]
    ds_train.train_spm(paths)
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    # Reverse
    src = 'tgt'
    tgt = 'src'
    model = 'transformer'
    criterion = 'cross_entropy'
    save_dir = root / f'models/{exp_id}_reverse'
    results_path = root / f'results/{exp_id}_reverse.txt'
    preprocess_dir = str(ds_train.data_dir) + '_reverse'
    Path(preprocess_dir).mkdir(exist_ok=True)

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {preprocess_dir} --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers 3 --decoder-layers 3 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 2 --decoder-attention-heads 2 \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 400 --warmup-init-lr 5e-4 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 500 --save-interval 500 --save-dir {save_dir} --user-dir {USER_DIR}"
    subprocess.run(shlex.split(cmd))

    srcdict = Path(preprocess_dir) / 'dict.src.txt'
    paths = [denoise_ds_train.src_path, denoise_ds_train.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    denoise_ds_train.src_bpe_path, denoise_ds_train.tgt_bpe_path = out_paths
    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(denoise_ds_train.src_bpe_path)} --destdir {denoise_ds_train.data_dir} --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate \
            {denoise_ds_train.data_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset train \
        --path {save_dir}/checkpoint_best.pt \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    src = 'src'
    tgt = 'tgt'
    reverse_save_dir = save_dir
    save_dir = root / f'models/{exp_id}'
    reverse_results_path = results_path
    results_path = root / f'results/{exp_id}.txt'
    preprocess_dir = Path(str(ds_train.data_dir) + '_forward')
    if preprocess_dir.exists():
        shutil.rmtree(preprocess_dir, ignore_errors=True)
    preprocess_dir.mkdir(exist_ok=False)

    # encode and preprocess the forward data
    pseudo_inputs_file = Path(parse_pred_file(reverse_results_path))
    pseudo_outputs_file = Path(denoise_ds_train.tgt_path)
    labeled_inputs_file = Path(ds_train.src_path)
    labeled_outputs_file = Path(ds_train.tgt_path)

    src_path = preprocess_dir / labeled_inputs_file.name
    tgt_path = preprocess_dir / labeled_outputs_file.name

    # copy over to preprocess_dir
    shutil.copyfile(labeled_inputs_file, src_path)
    shutil.copyfile(labeled_outputs_file, tgt_path)
    with open(src_path, 'a') as f:
        with open(pseudo_inputs_file, 'r') as f2:
            for line in f2:
                f.write(line.strip() + '\n')
    with open(tgt_path, 'a') as f:
        with open(pseudo_outputs_file, 'r') as f2:
            for line in f2:
                f.write(line.strip() + '\n')

    # concatenate with the labeled dataset
    paths = [src_path, tgt_path]
    encoded_paths = encode(paths, ds_train.spm_model_prefix)

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(encoded_paths[0])} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {preprocess_dir} --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # --label-smoothing 0.2 \
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers 3 --decoder-layers 3 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 2 --decoder-attention-heads 2 \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 400 --warmup-init-lr 5e-4 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 50 --save-interval 50 --save-dir {save_dir} --user-dir {USER_DIR}"
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate \
            {ds_train.data_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {save_dir}/checkpoint_best.pt \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id
    stats.append(res)


    # EVAL SECOND STEP
    cleaned_pred_path = str(root / f'results/{exp_id}_cleaned.txt')
    eval_dir = ds_train.data_dir / 'evaluate_2step_bt'
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(exist_ok=False)
    eval_src_path = eval_dir / 'testp.src'
    eval_tgt_path = eval_dir / 'testp.tgt'
    shutil.copy(str(cleaned_pred_path), eval_src_path)
    if not eval_tgt_path.exists():
        shutil.copy(ds_test.tgt_path, eval_tgt_path)

    # sentencepiece
    paths = [eval_src_path, eval_tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)

    srcdict = ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {strip_ext(out_paths[0])} --destdir {eval_dir} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    num_unlabeled = 20000
    denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    results_path_2 = results_path.parent / f"{results_path.stem}_pi.txt"
    # EVALUATE
    cmd = f"fairseq-generate {eval_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {denoise_save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path_2, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path_2, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id + ' (bt + denoise)'
    stats.append(res)



def composed_with_bt(num_examples=1000, eval_only=False, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'composed_with_bt_{ds_train.id}_dropout{dropout}'
    num_unlabeled = 20000
    denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
    std_exp_id = f'backtranslation_synthetic_{ds_train.id}_dropout{dropout}'

    # augmented training set
    preprocess_dir = Path(str(ds_train.data_dir) + '_forward')
    if not preprocess_dir.exists():
        raise ValueError("Run backtranslation first")

    src = 'src'
    tgt = 'tgt'
    model = 'double_transformer'
    criterion = 'reinforce_criterion'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{exp_id}.txt'
    results_path_2 = root / f'results/{exp_id}_pi.txt'

    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    standard_save_path = root / f'models/{std_exp_id}/checkpoint_best.pt'

    weight_decay = 0.0001

    ce_loss_lambda = 1.0

    if not eval_only:
        cmd = f"fairseq-train \
                {preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers 3 \
               --decoder-layers 3 \
               --encoder-embed-dim 256 \
               --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 \
               --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 2 \
               --decoder-attention-heads 2 \
               --encoder-layers-2 5 \
               --decoder-layers-2 5 \
               --encoder-embed-dim-2 256 \
               --decoder-embed-dim-2 256 \
               --encoder-ffn-embed-dim-2 1024 \
               --decoder-ffn-embed-dim-2 1024 \
               --encoder-attention-heads-2 4 \
               --decoder-attention-heads-2 4 \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay {weight_decay} \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 3000 --warmup-init-lr 1e-7 \
               --lr 2e-5 --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch 50 --save-interval 50 --save-dir {save_dir} --user-dir {USER_DIR} \
               --keep-best-checkpoints 1 \
               --pi-restore-path {denoise_save_path} \
               --f-restore-path {standard_save_path} \
               --ce-loss-lambda {ce_loss_lambda} \
               --best-checkpoint-metric loss_2 \
               --label-smoothing 0.0 --sampling"
        subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate \
            {str(ds_test.data_dir)} \
        --source-lang {src} --target-lang {tgt} \
        --path {save_dir}/checkpoint_best.pt \
        --gen-subset test \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id + " (base predictor)"
    stats.append(res)

    cleaned_pred_path = str(root / f'results/{exp_id}_cleaned.txt')
    eval_dir = ds_train.data_dir / 'evaluate_2step_composedwithbt'
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(exist_ok=False)
    eval_src_path = eval_dir / 'testp.src'
    eval_tgt_path = eval_dir / 'testp.tgt'
    shutil.copy(str(cleaned_pred_path), eval_src_path)
    if not eval_tgt_path.exists():
        shutil.copy(ds_test.tgt_path, eval_tgt_path)

    # sentencepiece
    paths = [eval_src_path, eval_tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)

    srcdict = ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {strip_ext(out_paths[0])} --destdir {eval_dir} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate {eval_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {denoise_save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR}"
    with open(results_path_2, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    res = evaluate_synthetic(pred=results_path_2, inputs=ds_test.inp_path,
                             gold=ds_test.tgt_path, gold_outputs=ds_test.data_dir / 'test.out')
    res['exp_id'] = exp_id
    stats.append(res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='only evaluate')
    parser.add_argument('--num_examples', type=int, default=1000,
                        help='num examples')
    parser.add_argument('--num_unlabeled', type=int, default=20000,
                        help='num unlabeled examples')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--suffix', type=str,
                        help='suffixes denote different instances of the synthetic dataset', default='a')
    args = parser.parse_args()

    GLOBAL_SUFFIX = args.suffix

    stats = []

    print("################### DIRECT ###################")
    direct(num_examples=args.num_examples, dropout=args.dropout)
    print("################### DENOISER ###################")
    denoise(num_examples=args.num_examples, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
    print("################### COMPOSED ONLY ###################")
    composed(num_examples=args.num_examples, pretrain_init=False, eval_only=args.eval_only, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
    print("################### PRETRAINED ###################")
    pretrained(num_examples=args.num_examples, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
    print("################### PRETRAINED + COMPOSED ###################")
    composed(num_examples=args.num_examples, pretrain_init=True, eval_only=args.eval_only, num_unlabeled=args.num_unlabeled, dropout=args.dropout)

    if args.num_unlabeled == 20000:
        print("################### BACKTRANSLATION ###################")
        backtranslation(num_examples=args.num_examples, dropout=args.dropout)
        print("################### BACKTRANSLATION + COMPOSED ###################")
        composed_with_bt(num_examples=args.num_examples, dropout=args.dropout)

    res = pd.DataFrame(stats)
    res = res.round(4)
    res.to_csv('synthetic_results.tsv', sep='\t', index=None)
    print(res)
