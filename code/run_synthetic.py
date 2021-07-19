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

    def __init__(self, root, split='train', denoise=False, num_examples=1000, num_unlabeled=20000, denoise_random=False):
        self.split = split
        self.root = Path(root)
        self.id = []
        if denoise:
            self.id.append('denoise')
        if denoise_random:
            self.id.append('denoise_random')
        if num_unlabeled != 20000:
            self.id.append(f'numunlabeled{num_unlabeled}')

        self.id.append(str(num_examples))
        self.id = '_'.join(self.id)
        self.id += GLOBAL_SUFFIX

        self.data_dir = self.root / f'synthetic_{self.id}'
        self.denoise = denoise
        self.denoise_random = denoise_random
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

    def random_delete(self, code_str):
        toks = np.asarray(code_str.split())
        mask = np.random.rand(len(toks)) > 0.15
        return ' '.join(toks[mask])

    def get_print_subtype(self, out_str):
        code = f"cout << {out_str}; "

        if self.split == 'ood':
            pseudocode = np.random.choice([
                    f"print {out_str} to stdout; ",
                    f"output {out_str}; ",
                    f"stdout {out_str}; "])
        else:
            pseudocode = np.random.choice([
                    f"print {out_str}; ",
                    f"output {out_str} to stdout; "])

        if self.denoise_random:
            noised_code = self.random_delete(code)
        else:
            noised_code = ""

        return "", pseudocode, code, noised_code

    def get_input_subtype(self, var_list, type_list):
        var_list_str = ' >> '.join(var_list)
        code = f"cin >> {var_list_str}; "

        listed_var_list_str = ', '.join(var_list)
        if self.denoise_random:
            noised_code = self.random_delete(code)
        else:
            noised_code = np.random.choice([
                f"cin {var_list_str}; ",
                f"cin >> {listed_var_list_str}; ",
                f"cin << {var_list_str}; "])

        if self.split == 'ood':
            pseudocode = np.random.choice([
                f"read {listed_var_list_str}; ",
                f"{listed_var_list_str} from stdin; ",])
        else:
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
            if self.denoise_random:
                noised_code = self.random_delete(code)
            else:
                noised_code = np.random.choice([
                            f"{np.random.choice(other_types)} {var_str} = {value}; ",
                            f"{var_str} = {value}; ",
                            f"{np.random.choice(other_types)} {var_str}; {var_str} = {value}; ",
                            ])
        else:
            code = f"{var_str} = {value}; "
            if self.denoise_random:
                noised_code = self.random_delete(code)
            else:
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
        if self.denoise_random:
            noised_code = self.random_delete(code)
        else:
            noised_code = f"{np.random.choice(other_types)} {', '.join(var_list)}; "
        return "", pseudocode, code, noised_code

    def get_type_preserving_subtype(self, var_str, type_str, curr_vars):
        # ambiguity in add and set
        if type_str == 'int':
            value = self.get_value(type_str)
            type_idx = np.random.randint(3)
            if type_idx == 0:
                code = np.random.choice([f'{var_str} += {value}; ', f'{var_str} = {var_str} + {value}; '])
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                            f'set {var_str} to {var_str} plus {value}; ',
                            f'{var_str} to {value} plus {var_str}; ',
                            f'set {var_str} to {value} plus the value of {var_str}; ', ])
                else:
                    pseudocode = np.random.choice([
                            f'set {var_str} to {value} plus {var_str}; '])
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
                    noised_code = np.random.choice([
                                f'{type_str} {var_str} += {value}; ',
                                f'{type_str} {var_str} = {var_str} + {value}; ',
                                f'{np.random.choice(self.types)} {var_str} += {value}; ',
                                f'{np.random.choice(self.types)} {var_str} = {var_str} + {value}; '])
            elif type_idx == 1:
                code = np.random.choice([f'{var_str} -= {value}; ', f'{var_str} = {var_str} - {value}; '])
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                            f'{var_str} to {value} minus {var_str}; ',
                            f'set {var_str} to {value} minus the value of {var_str}; ', ])
                else:
                    pseudocode = np.random.choice([
                            f'set {var_str} to {value} minus {var_str}; '])
                other_var = self.get_var()
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
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
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                        f'negation of {var_str}; ',
                        f'set {var_str} to negation; ',
                        f'set {var_str} to negation of {var_str}; ',
                        ])
                else:
                    pseudocode = f'set {var_str} to its negation; '
                code = f'{var_str} = !{var_str}; '
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
                    noised_code = np.random.choice([
                       f'{type_str} {var_str} = !{var_str}; ',
                       f'{np.random.choice(self.types)} {var_str} = !{var_str}; '])
            if type_idx == 0:
                code = np.random.choice([
                    f'{var_str} = {var_str} && {value}; '])
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                        f'set {var_str} to {var_str} and {value}; ',
                        f'and {var_str} with {value}; ',
                        f'{var_str} and {value}; ',
                        ])
                else:
                    pseudocode = f'set {var_str} to the and with {value}; '
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
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
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                        f"if {var_str} true, ",
                        f"if {var_str}, ",
                        ])
                else:
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

                if self.denoise_random:
                    noised_code = self.random_delete(code)

                if self.split == 'ood':
                    pseudocode += np.random.choice([
                        f'set {var_1} to {var_2} and set {var_2} to {var_1}; ',
                        f'set {var_1} to the value of {var_2} and {var_2} to {var_1}; ',
                        f'swap {var_1} and {var_2}; '])
                else:
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
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                        f'add {value} to {var_str} end; ',
                        f'add {value} to end of {var_str}; ',])
                else:
                    pseudocode = f'add {value} to the end of {var_str}; '
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
                    noised_code = np.random.choice([
                            f'{np.random.choice(self.types)} {var_str} = {var_str} + {value}; ',
                            f"{np.random.choice(self.types)} {var_str} += {value}; ",
                            f'{type_str} {var_str} = {var_str} + {value}; ',
                            f"{type_str} {var_str} += {value}; "])
            elif type_idx == 1:
                code = f'{var_str} = {value} + {var_str}; '
                if self.split == 'ood':
                    pseudocode = np.random.choice([
                        f'add {value} to {var_str} beginning; ',
                        f'add {value} to beginning of {var_str}; ',])
                else:
                    pseudocode = f'add {value} to the beginning of {var_str}; '
                if self.denoise_random:
                    noised_code = self.random_delete(code)
                else:
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
        pseudocode = f"set {var_set} to {func} applied to {var_str1} and {var_str2}; "

        code = f"{var_set} = {func}({var_str1}, {var_str2}); "
        if self.denoise_random:
            noised_code = self.random_delete(code)
        else:
            noised_code = f"{np.random.choice(self.types)} {var_set} = {func}({var_str1}, {var_str2}); "
        return (
            "",
            pseudocode,
            code,
            noised_code,
            )

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
                if self.denoise_random:
                    # just use all noisy versions
                    noised_code_ret = list(noised_code)
                else:
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


def eval_for_ds(ds, exp_id, data_dir, ds_train, src, tgt, save_dir, results_path, gold_outputs_filename='test.out', eval_mode='all'):
    if eval_mode in {'all','first'}:
        # EVALUATE
        cmd = f"fairseq-generate \
                {data_dir} \
            --source-lang {src} --target-lang {tgt} \
            --gen-subset test \
            --path {save_dir}/checkpoint_best.pt \
            --beam 1 \
            --remove-bpe=sentencepiece --user-dir {USER_DIR}"
        with open(results_path, 'w') as f:
            subprocess.run(shlex.split(cmd), stdout=f)

        res = evaluate_synthetic(pred=results_path, inputs=ds.inp_path,
                                 gold=ds.tgt_path, gold_outputs=ds.data_dir / gold_outputs_filename)
        res['exp_id'] = exp_id
        stats.append(res)


    if eval_mode in {'all', 'second'}:
        # EVAL SECOND STEP
        cleaned_pred_path = str(save_root / f'results/{exp_id}_cleaned.txt')
        eval_dir = save_root / 'intermediate' / f'evaluate_2step_{exp_id}'
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(exist_ok=False)
        eval_src_path = eval_dir / 'testp.src'
        eval_tgt_path = eval_dir / 'testp.tgt'
        shutil.copy(str(cleaned_pred_path), eval_src_path)
        if not eval_tgt_path.exists():
            shutil.copy(ds.tgt_path, eval_tgt_path)

        # sentencepiece
        paths = [eval_src_path, eval_tgt_path]
        out_paths = encode(paths, ds_train.spm_model_prefix)

        srcdict = data_dir / f'dict.{src}.txt'

        cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
                --testpref {strip_ext(out_paths[0])} --destdir {eval_dir} \
                --joined-dictionary --workers 2 --srcdict {srcdict}'
        subprocess.run(shlex.split(cmd))

        num_unlabeled = 20000
        if args.denoise_random:
            denoise_exp_id = f'denoise_random_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
        else:
            denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
        if NUM_LAYERS != 3:
            denoise_exp_id += f'_{NUM_LAYERS}'
        denoise_save_path = save_root / f'models/{denoise_exp_id}/checkpoint_best.pt'
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

        res = evaluate_synthetic(pred=results_path_2, inputs=ds.inp_path,
                                 gold=ds.tgt_path, gold_outputs=ds.data_dir / gold_outputs_filename)
        res['exp_id'] = exp_id + ' + denoise'
        stats.append(res)


def evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all'):
    eval_for_ds(ds_test, exp_id, ds_train.data_dir, ds_train, src, tgt, save_dir, results_path, eval_mode=eval_mode)

    root = Path(ROOT)

    ds_ood = Synthetic(root / 'data', split='ood', num_examples=num_examples)
    paths = [ds_ood.src_path, ds_ood.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_ood.src_bpe_path, ds_ood.tgt_bpe_path = out_paths
    results_path_ood = Path(str(strip_ext(results_path)) + '_ood.txt')
    data_dir = Path(str(ds_train.data_dir) + '_ood')
    data_dir.mkdir(exist_ok=True)
    srcdict = ds_train.data_dir / f'dict.{src}.txt'
    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --testpref {strip_ext(ds_ood.src_bpe_path)} --destdir {data_dir} --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))
    eval_for_ds(ds_ood, exp_id + '_ood', data_dir, ds_train, src, tgt, save_dir, results_path_ood, eval_mode=eval_mode, gold_outputs_filename='ood.out')


def direct(num_examples=1000, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'direct_synthetic_{ds_train.id}_dropout{dropout}'

    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'

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
    save_dir = save_root / f'models/{exp_id}'
    results_path = save_root / f'results/{exp_id}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2'
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-train \
            {ds_train.data_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
           --max-epoch 500 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR}"
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='first')


def denoise(num_examples=1000, do_eval=True, num_unlabeled=20000, dropout=0.1):
    root = Path(ROOT)
    std_ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    std_ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    std_ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)
    ds_train = Synthetic(root / 'data', split='train', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled, denoise_random=args.denoise_random)
    ds_val = Synthetic(root / 'data', split='val', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled, denoise_random=args.denoise_random)
    ds_test = Synthetic(root / 'data', split='test', denoise=True, num_examples=num_examples, num_unlabeled=num_unlabeled, denoise_random=args.denoise_random)

    if args.denoise_random:
        exp_id = f'denoise_random_synthetic_{std_ds_train.id}_numunlabeled{num_unlabeled}'
    else:
        exp_id = f'denoise_synthetic_{std_ds_train.id}_numunlabeled{num_unlabeled}'
    std_exp_id = f'direct_synthetic_{std_ds_train.id}_dropout{dropout}'

    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'
        std_exp_id += f'_{NUM_LAYERS}'

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
    save_dir = save_root / f'models/{exp_id}'
    results_path = save_root / f'results/{exp_id}.txt'

    srcdict = std_ds_train.data_dir / f'dict.{src}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2 --srcdict {srcdict}'
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    params = f"--encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 2 --decoder-attention-heads 2 "

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
           --max-epoch 50 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR}"
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    if do_eval:
        std_results_path = save_root / f'results/{std_exp_id}.txt'
        evaluate_allsteps(std_exp_id, std_ds_train, src, tgt, save_dir, std_results_path, std_ds_test, num_examples, eval_mode='second')
    return exp_id


def composed(num_examples=1000, pretrain_init=False, eval_only=False, num_unlabeled=20000, dropout=0.1, bartstyle_init=False):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    if pretrain_init:
        exp_id = f'composed_synthetic_pretrain_{ds_train.id}'
    elif bartstyle_init:
        exp_id = f'composed_synthetic_pretrain_{ds_train.id}_bartstyle'
    else:
        exp_id = f'composed_synthetic_{ds_train.id}'

    if args.ce_loss_lambda != 1.0:
        exp_id += f'_lambda{args.ce_loss_lambda}'
    if args.no_init:
        exp_id += '_noinit'

    if args.denoise_random:
        exp_id += 'denoise_random'

    if args.denoise_random:
        denoise_exp_id = f'denoise_random_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
    else:
        denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'

    suffix = f'_numunlabeled{num_unlabeled}_dropout{dropout}'
    std_suffix = f'_dropout{dropout}'
    exp_id += suffix

    if pretrain_init or bartstyle_init:
        std_exp_id = f'pretrain_synthetic_{ds_train.id}'
        std_exp_id += suffix
        if args.denoise_random:
            std_exp_id += 'denoise_random'
    else:
        std_exp_id = f'direct_synthetic_{ds_train.id}'
        std_exp_id += std_suffix

    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'
        denoise_exp_id += f'_{NUM_LAYERS}'
        std_exp_id += f'_{NUM_LAYERS}'

    if bartstyle_init:
        std_exp_id += '_bartstyle'

    src = 'src'
    tgt = 'tgt'
    model = 'double_transformer'
    criterion = 'reinforce_criterion'

    save_dir = save_root / f'models/{exp_id}'
    results_path = save_root / f'results/{exp_id}.txt'
    results_path_2 = save_root / f'results/{exp_id}_pi.txt'

    denoise_save_path = save_root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    standard_save_path = save_root / f'models/{std_exp_id}/checkpoint_best.pt'

    weight_decay = 0.0001
    ce_loss_lambda = args.ce_loss_lambda

    lr = 2e-4

    if not eval_only:
        cmd = f"fairseq-train \
                {ds_train.data_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {NUM_LAYERS} \
               --decoder-layers {NUM_LAYERS} \
               --encoder-embed-dim 256 \
               --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 \
               --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 2 \
               --decoder-attention-heads 2 \
               --encoder-layers-2 {NUM_LAYERS} \
               --decoder-layers-2 {NUM_LAYERS} \
               --encoder-embed-dim-2 256 \
               --decoder-embed-dim-2 256 \
               --encoder-ffn-embed-dim-2 1024 \
               --decoder-ffn-embed-dim-2 1024 \
               --encoder-attention-heads-2 2 \
               --decoder-attention-heads-2 2 \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay {weight_decay} \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 90 --warmup-init-lr 1e-7 \
               --lr {lr} --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch 300 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR} \
               --pi-restore-path {denoise_save_path} \
               --ce-loss-lambda {ce_loss_lambda} \
               --label-smoothing 0.0 --sampling "
        if not args.no_init:
            cmd += f" --f-restore-path {standard_save_path} "
        cmd += " --best-checkpoint-metric loss_2 "
        subprocess.run(shlex.split(cmd))

    # note: composed returns a model where the forward defaults to the base predictor only
    evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all')


def pretrained(num_examples=1000, num_unlabeled=20000, dropout=0.1, bart_style_finetune=False):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'pretrain_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}_dropout{dropout}'

    if args.denoise_random:
        denoise_exp_id = f'denoise_random_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
    else:
        denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'

    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'
        denoise_exp_id += f'_{NUM_LAYERS}'

    paths = [ds_train.src_path, ds_train.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    if bart_style_finetune:
        exp_id += '_bartstyle'
    if args.denoise_random:
        exp_id += 'denoise_random'

    src = 'src'
    tgt = 'tgt'
    criterion = 'cross_entropy'
    save_dir = save_root / f'models/{exp_id}'
    denoise_save_path = save_root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    results_path = save_root / f'results/{exp_id}.txt'
    results_path_2 = save_root / f'results/{exp_id}_pi.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {ds_train.data_dir} --joined-dictionary --workers 2'
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    if not bart_style_finetune:
        model = 'transformer'
        cmd = f"fairseq-train \
                {ds_train.data_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
               --max-epoch 300 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR} \
               --restore-file {denoise_save_path} \
               --reset-optimizer \
               --reset-lr-scheduler \
               --reset-dataloader \
               --reset-meters"
        if not args.eval_only:
            subprocess.run(shlex.split(cmd))

        evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all')
    else:
        model = 'custom_transformer'
        cmd = f"fairseq-train \
                {ds_train.data_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
               --max-epoch 200 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR} \
               --f-restore-path {denoise_save_path}"
        if not args.eval_only:
            subprocess.run(shlex.split(cmd))

        cmd = f"fairseq-train \
                {ds_train.data_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
               --max-epoch 300 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR} \
               --reset-optimizer \
               --reset-lr-scheduler \
               --f-restore-path {save_dir / 'checkpoint_best.pt'} \
               --train-all"
        if not args.eval_only:
            subprocess.run(shlex.split(cmd))

        evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all')


def backtranslation(num_examples=1000, dropout=0.1, forward_only=True):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)
    denoise_ds_train = Synthetic(root / 'data', split='train', denoise=True, num_examples=num_examples)

    exp_id = f'backtranslation_synthetic_{ds_train.id}_dropout{dropout}'
    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'

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
    save_dir = save_root / f'models/{exp_id}_reverse'
    results_path = save_root / f'results/{exp_id}_reverse.txt'
    preprocess_dir = str(ds_train.data_dir) + '_reverse'
    if NUM_LAYERS != 3:
        preprocess_dir += f'_{NUM_LAYERS}'
    Path(preprocess_dir).mkdir(exist_ok=True)

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} --trainpref {strip_ext(ds_train.src_bpe_path)} --validpref {strip_ext(ds_val.src_bpe_path)} --testpref {strip_ext(ds_test.src_bpe_path)} --destdir {preprocess_dir} --joined-dictionary --workers 2'
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
           --max-epoch 500 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR}"
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    srcdict = Path(preprocess_dir) / 'dict.src.txt'
    paths = [denoise_ds_train.src_path, denoise_ds_train.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    denoise_ds_train.src_bpe_path, denoise_ds_train.tgt_bpe_path = out_paths
    if not args.eval_only or not forward_only:
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
    save_dir = save_root / f'models/{exp_id}'
    reverse_results_path = results_path
    results_path = save_root / f'results/{exp_id}.txt'
    preprocess_dir = str(ds_train.data_dir) + '_forward'
    if NUM_LAYERS != 3:
        preprocess_dir += f'_{NUM_LAYERS}'
    preprocess_dir = Path(preprocess_dir)
    if not args.eval_only or not forward_only:
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

    if not args.eval_only or not forward_only:
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
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    # --label-smoothing 0.2 \
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           --encoder-layers {NUM_LAYERS} --decoder-layers {NUM_LAYERS} \
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
           --max-epoch 300 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR}"
    if not args.eval_only:
        subprocess.run(shlex.split(cmd))

    evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all')


def composed_with_bt(num_examples=1000, eval_only=False, dropout=0.1):
    root = Path(ROOT)
    ds_train = Synthetic(root / 'data', split='train', num_examples=num_examples)
    ds_val = Synthetic(root / 'data', split='val', num_examples=num_examples)
    ds_test = Synthetic(root / 'data', split='test', num_examples=num_examples)

    exp_id = f'composed_with_bt_{ds_train.id}_dropout{dropout}'
    if args.ce_loss_lambda != 1.0:
        exp_id += f'_lambda{args.ce_loss_lambda}'
    if args.no_init:
        exp_id += '_noinit'
    num_unlabeled = 20000
    denoise_exp_id = f'denoise_synthetic_{ds_train.id}_numunlabeled{num_unlabeled}'
    std_exp_id = f'backtranslation_synthetic_{ds_train.id}_dropout{dropout}'

    if NUM_LAYERS != 3:
        exp_id += f'_{NUM_LAYERS}'
        std_exp_id += f'_{NUM_LAYERS}'
        denoise_exp_id += f'_{NUM_LAYERS}'

    # augmented training set
    preprocess_dir = Path(str(ds_train.data_dir) + '_forward')
    if not preprocess_dir.exists():
        raise ValueError("Run backtranslation first")

    src = 'src'
    tgt = 'tgt'
    model = 'double_transformer'
    criterion = 'reinforce_criterion'
    save_dir = save_root / f'models/{exp_id}'
    results_path = save_root / f'results/{exp_id}.txt'
    results_path_2 = save_root / f'results/{exp_id}_pi.txt'

    denoise_save_path = save_root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    standard_save_path = save_root / f'models/{std_exp_id}/checkpoint_best.pt'

    weight_decay = 0.0001

    ce_loss_lambda = args.ce_loss_lambda

    if not eval_only:
        cmd = f"fairseq-train \
                {preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               --encoder-layers {NUM_LAYERS} \
               --decoder-layers {NUM_LAYERS} \
               --encoder-embed-dim 256 \
               --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 \
               --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 2 \
               --decoder-attention-heads 2 \
               --encoder-layers-2 {NUM_LAYERS} \
               --decoder-layers-2 {NUM_LAYERS} \
               --encoder-embed-dim-2 256 \
               --decoder-embed-dim-2 256 \
               --encoder-ffn-embed-dim-2 1024 \
               --decoder-ffn-embed-dim-2 1024 \
               --encoder-attention-heads-2 2 \
               --decoder-attention-heads-2 2 \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout {dropout} --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay {weight_decay} \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 3000 --warmup-init-lr 1e-7 \
               --lr 2e-5 --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch 50 --save-interval 99999 --save-dir {save_dir} --user-dir {USER_DIR} \
               --keep-best-checkpoints 1 \
               --pi-restore-path {denoise_save_path} \
               --ce-loss-lambda {ce_loss_lambda} \
               --label-smoothing 0.0 --sampling "
        if not args.no_init:
            cmd += f" --f-restore-path {standard_save_path} "
        cmd += " --best-checkpoint-metric loss_2 "

        subprocess.run(shlex.split(cmd))

    evaluate_allsteps(exp_id, ds_train, src, tgt, save_dir, results_path, ds_test, num_examples, eval_mode='all')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scripts')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='only evaluate')
    parser.add_argument('--num_examples', type=int, default=1000,
                        help='num examples')
    parser.add_argument('--num_unlabeled', type=int, default=20000,
                        help='num unlabeled examples')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--suffix', type=str,
                        help='suffixes denote different instances of the synthetic dataset', default='a')
    parser.add_argument('--mode', type=str,
                        help='which algorithm to run', default='direct')
    parser.add_argument('--ce_loss_lambda', type=float, default=1.0, help='hyperparmeter in objective function')
    parser.add_argument('--no_init', action='store_true', help='dont initialize from something else')
    parser.add_argument('--full_bt', action='store_true', help='run all backtranslation, including training reverse model')
    parser.add_argument('--denoise_random', action='store_true', help='random deletions for noise')
    parser.add_argument('--save_root', type=str,
                        help='where to save inside root', default='.')
    args = parser.parse_args()

    save_root = Path(ROOT) / Path(args.save_root)
    for dname in ['intermediate', 'results', 'models']:
        d = save_root / dname
        d.mkdir(exist_ok=True)

    GLOBAL_SUFFIX = args.suffix

    NUM_LAYERS = args.num_layers

    stats = []

    if ',' in args.mode:
        mode_list = args.mode.split(',')
    else:
        mode_list = [args.mode]

    for mode in mode_list:
        if mode == 'direct':
            print("################### DIRECT ###################")
            direct(num_examples=args.num_examples, dropout=args.dropout)
        elif mode == 'denoise':
            print("################### DENOISER ###################")
            denoise(num_examples=args.num_examples, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
        elif mode == 'composed-stdinit':
            print("################### COMPOSED ONLY ###################")
            composed(num_examples=args.num_examples, pretrain_init=False, eval_only=args.eval_only, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
        elif mode == 'pretrain':
            print("################### PRETRAINED ###################")
            pretrained(num_examples=args.num_examples, num_unlabeled=args.num_unlabeled, dropout=args.dropout)
        elif mode == 'pretrain+compose':
            print("################### PRETRAINED + COMPOSED ###################")
            composed(num_examples=args.num_examples, pretrain_init=True, eval_only=args.eval_only, num_unlabeled=args.num_unlabeled, dropout=args.dropout)

        elif mode == 'backtranslation'  and args.num_unlabeled == 20000:
            print("################### BACKTRANSLATION ###################")
            backtranslation(num_examples=args.num_examples, dropout=args.dropout, forward_only=(not args.full_bt))
        elif mode == 'backtranslation+compose' and args.num_unlabeled == 20000:
            print("################### BACKTRANSLATION + COMPOSED ###################")
            composed_with_bt(num_examples=args.num_examples, dropout=args.dropout, eval_only=args.eval_only)
        elif mode == 'pretrain_bartstyle':
            print("################### PRETRAINED BART-STYLE FINETUNE ###################")
            pretrained(num_examples=args.num_examples, num_unlabeled=args.num_unlabeled, dropout=args.dropout, bart_style_finetune=True)
        elif mode == 'pretrain+compose_bartstyle':
            print("################### PRETRAINED + COMPOSED BART STYLE ###################")
            composed(num_examples=args.num_examples, pretrain_init=False, eval_only=args.eval_only, num_unlabeled=args.num_unlabeled, dropout=args.dropout, bartstyle_init=True)
        else:
            raise ValueError("mode not supported")

    cols = ['exp_id', 'comp_err', 'mismatch_err', 'no_err']
    res = pd.DataFrame(stats)
    res = res[cols]
    res = res.round(4)
    Path(save_root / 'result_tables').mkdir(exist_ok=True)
    res.to_csv(save_root / f'result_tables/synthetic_results_{NUM_LAYERS}_{args.mode}_nexamples{args.num_examples}_nunlabeled{args.num_unlabeled}.tsv', sep='\t', index=None)
    print(res)
