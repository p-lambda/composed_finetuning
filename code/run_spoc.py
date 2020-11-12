import argparse
import subprocess
import shlex
import logging
import requests
import pandas as pd
import shutil
from pathlib import Path
from zipfile import ZipFile

from torch.utils.data import Dataset
from tqdm import tqdm
import sentencepiece as spm

from c_tokenizer_mod import C_Tokenizer
from evaluate_full import evaluate_full_spoc

USER_DIR = 'fairseq_modules/'
ROOT = '.'


def strip_ext(path):
    return path.parent / path.stem


def concat_code(curr_df, tokenizer):
    code_list = []
    for i in range(len(curr_df)):
        curr_obj = curr_df.iloc[i]
        curr_code = ' '.join(tokenizer.tokenize(curr_obj.code)[0])
        curr_code = '~'*curr_obj.indent + curr_code + ' '
        code_list.append(curr_code)
    code = ' $ '.join(code_list)
    return code.strip()


def concat_text(curr_df):
    text_list = []
    for i in range(len(curr_df)):
        s = curr_df.iloc[i].text
        ind = curr_df.iloc[i].indent
        if type(s) == float:
            continue
        else:
            indents = f' TAB{ind} '
            text_list.append(indents + s + ' ; ')
    curr_text = ' $ '.join(text_list)
    return curr_text.strip()


class SPoC(Dataset):
    url = 'https://github.com/Sumith1896/spoc/raw/gh-pages/data/spoc.zip'
    base_folder = 'spoc-data'

    def __init__(self, root, transform=None, target_transform=None, split='train',
                 download=True, denoise=False, denoise_pre=False,
                 finetune_classifier=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.base = Path(self.root) / self.base_folder
        self.denoise = denoise
        self.denoise_pre = denoise_pre
        self.finetune_classifier = finetune_classifier
        self.tokenizer = C_Tokenizer()

        if download:
            self.download()

        self.data_dir = self.base / 'train' / 'split'
        if self.denoise:
            self.preprocess_denoise(denoise_pre, finetune_classifier)
        else:
            self.preprocess()

    def __len__(self):
        return len(self.idxs)

    def preprocess_denoise(self, denoise_pre=False, balanced=False):
        self.preprocess_dir = self.data_dir / 'preprocess_fairseq_denoise'

        if denoise_pre:
            self.preprocess_dir = Path(str(self.preprocess_dir) + '_linewise')

        if balanced:
            self.preprocess_dir = Path(str(self.preprocess_dir) + '_classifier')

        self.preprocess_dir.mkdir(exist_ok=True)
        self.src_path = self.preprocess_dir / f'spoc-train-{self.split}_preprocess.src'
        self.tgt_path = self.preprocess_dir / f'spoc-train-{self.split}_preprocess.tgt'

    def preprocess(self):
        self.preprocess_dir = self.data_dir / 'preprocess_fairseq'

        if self.split in {'testp', 'testw'}:
            self.preprocess_dir = self.preprocess_dir / f'evaluate_{self.split}'
        self.preprocess_dir.mkdir(exist_ok=True)

        if self.split == 'val':
            # hack for the different naming convention
            input_path = self.data_dir / 'spoc-train-eval.tsv'
        elif self.split == 'test':
            input_path = self.data_dir / 'spoc-train-test.tsv'
        elif self.split == 'train':
            input_path = self.data_dir / 'spoc-train-train.tsv'
        elif self.split == 'testp':
            input_path = self.base / 'test' / 'spoc-testp.tsv'
        elif self.split == 'testw':
            input_path = self.base / 'test' / 'spoc-testw.tsv'

        self.src_path = self.preprocess_dir / f'spoc-train-{self.split}_preprocess.src'
        self.tgt_path = self.preprocess_dir / f'spoc-train-{self.split}_preprocess.tgt'
        self.id_path = self.preprocess_dir / f'spoc-train-{self.split}_preprocess.id'

        if not self.src_path.exists() or not self.tgt_path.exists():
            logging.info("Preprocessing data")
            df = pd.read_csv(input_path, sep='\t')
            df["id"] = df["probid"] + "-" + df["subid"].astype(str) + "-" + df["workerid"].astype(str)

            texts = []
            codes = []
            ids = []
            for id in tqdm(list(df["id"].unique())):
                curr_df = df[df["id"] == id]
                curr_code = concat_code(curr_df, self.tokenizer)

                # filter out long examples
                if self.split not in {'testp', 'testw'} and len(curr_code) > 1000:
                    continue

                ids.append(id)
                curr_text = concat_text(curr_df)
                texts.append(curr_text)
                codes.append(curr_code)

            with open(self.id_path, 'w') as f:
                for id in ids:
                    f.write(id + '\n')

            with open(self.src_path, 'w') as f:
                for text in texts:
                    f.write(text.strip() + '\n')
            with open(self.tgt_path, 'w') as f:
                for code in codes:
                    f.write(code.strip() + '\n')

    def download(self):
        self.base.mkdir(exist_ok=True, parents=True)
        path = self.base / 'spoc.zip'

        if not path.exists():
            logging.info("Downloading dataset")
            r = requests.get(self.url)
            with open(path, 'wb') as f:
                f.write(r.content)

            with ZipFile(path) as z:
                z.extractall(path=self.base)

    def train_spm(self, paths, do_train=True):
        self.vocab_size = 10000
        self.spm_model_prefix = self.preprocess_dir / 'sentencepiece.bpe'
        if do_train:
            paths = [str(p) for p in paths]
            if not Path(f"{str(self.spm_model_prefix)}.model").exists() or not Path(f"{str(self.spm_model_prefix)}.vocab").exists():
                cmd = f"--input={','.join(paths)} \
                        --model_prefix={self.spm_model_prefix} \
                        --vocab_size={self.vocab_size} \
                        --character_coverage=1.0 \
                        --model_type=bpe"
                spm.SentencePieceTrainer.Train(cmd)


def encode(in_paths, spm_model_prefix):
    out_paths = [str(p.parent / f'{p.stem}.bpe{p.suffix}')
                 for p in in_paths]
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


def evaluate_spoc(save_path, results_path, encode, srcdict, root, exp_id, split='testp'):
    src = 'src'
    tgt = 'tgt'
    ds_test = SPoC(root / 'data', split=split)

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {str(strip_ext(ds_test.src_bpe_path))} \
            --destdir {str(ds_test.preprocess_dir)} --joined-dictionary \
            --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    cmd = f"fairseq-generate \
            {str(ds_test.preprocess_dir)} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR} \
        --max-source-positions 2048 --max-target-positions 2048"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    cmd = f"python evaluate_full.py \
            --id_path {ds_test.id_path} \
            --pred {str(results_path)}"
    subprocess.run(shlex.split(cmd))

    res = evaluate_full_spoc(id_path=ds_test.id_path, pred=results_path)
    res['exp_id'] = exp_id
    res['split'] = split
    stats.append(res)


def evaluate_spoc_round2(std_cleaned_pred_path, denoise_save_path, eval_dir, encode, srcdict, root, results_path_2, exp_id, do_evaluate_full=True, split='testp'):
    src = 'src'
    tgt = 'tgt'
    ds_test = SPoC(root / 'data', split=split)
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(exist_ok=False)
    eval_src_path = eval_dir / f'{split}.src'
    eval_tgt_path = eval_dir / f'{split}.tgt'
    shutil.copy(std_cleaned_pred_path, eval_src_path)
    shutil.copy(ds_test.tgt_path, eval_tgt_path)

    # sentencepiece
    paths = [eval_src_path, eval_tgt_path]
    out_paths = encode(paths)

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --testpref {str(strip_ext(out_paths[0]))} --destdir {str(eval_dir)} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))

    # EVALUATE
    cmd = f"fairseq-generate {eval_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset test \
        --path {denoise_save_path} \
        --beam 1 \
        --remove-bpe=sentencepiece --user-dir {USER_DIR} \
        --max-source-positions 2048 --max-target-positions 2048"
    with open(results_path_2, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    if do_evaluate_full:
        res = evaluate_full_spoc(id_path=ds_test.id_path, pred=results_path_2)
        if exp_id is not None:
            res['exp_id'] = exp_id
            res['split'] = split
            stats.append(res)


def direct():
    root = Path(ROOT)
    ds_train = SPoC(root / 'data', split='train')
    ds_val = SPoC(root / 'data', split='val')
    ds_test = SPoC(root / 'data', split='test')

    exp_id = 'direct_spoc'

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
    criterion = 'label_smoothed_cross_entropy'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{exp_id}.txt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {str(strip_ext(ds_train.src_bpe_path))} \
            --validpref {str(strip_ext(ds_val.src_bpe_path))} \
            --testpref {str(strip_ext(ds_test.src_bpe_path))} \
            --destdir {str(ds_train.preprocess_dir)} \
            --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    warmup_updates = 4000
    warmup_init_lr = 1e-7
    lr = 1e-3
    label_smoothing = 0.2
    max_epoch=100
    params = " --encoder-layers 5 --decoder-layers 5 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 8 --decoder-attention-heads 8 "
    cmd = f"fairseq-train \
            {ds_train.preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --label-smoothing {label_smoothing} --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
           --lr {lr} --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch {max_epoch} --save-interval 25 --save-dir {save_dir} --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))

    max_epoch += 50
    label_smoothing = 0.1
    cmd = f"fairseq-train \
            {ds_train.preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --label-smoothing {label_smoothing} --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
           --lr {lr} --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch {max_epoch} --save-interval 25 --save-dir {save_dir} --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))

    # EVALUATE

    # dict files
    srcdict = ds_train.preprocess_dir / f'dict.{src}.txt'
    tgtdict = ds_train.preprocess_dir / f'dict.{tgt}.txt'

    encode_fn = lambda paths: encode(paths, ds_train.spm_model_prefix)
    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testp', exp_id=exp_id)
    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testw', exp_id=exp_id)


def denoise(denoise_pre=False, use_denoise_pre=False, finetune_classifier=False, do_eval=False):
    if not denoise_pre and use_denoise_pre:
        # linewise pretraining
        denoise(denoise_pre=True, use_denoise_pre=False)

    root = Path(ROOT)
    std_ds_train = SPoC(root / 'data', split='train')
    std_ds_val = SPoC(root / 'data', split='val')
    std_ds_test = SPoC(root / 'data', split='test')
    ds_train = SPoC(root / 'data', split='train', denoise=True, denoise_pre=denoise_pre, finetune_classifier=finetune_classifier)
    ds_val = SPoC(root / 'data', split='val', denoise=True, denoise_pre=denoise_pre, finetune_classifier=finetune_classifier)
    ds_test = SPoC(root / 'data', split='test', denoise=True, denoise_pre=denoise_pre, finetune_classifier=finetune_classifier)

    exp_id = 'denoise_spoc'
    std_exp_id = 'direct_spoc'
    pre_exp_id = 'denoise_spoc'
    denoise_exp_id = 'denoise_spoc'

    if denoise_pre:
        exp_id += '_linewise'

    if use_denoise_pre:
        pre_exp_id += '_linewise'

    if finetune_classifier:
        exp_id += '_classifier'

    src = 'src'
    tgt = 'tgt'
    # train and encode spm
    std_ds_train.spm_model_prefix = std_ds_train.preprocess_dir / 'sentencepiece.bpe'

    paths = [ds_train.src_path, ds_train.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_train.src_bpe_path, ds_train.tgt_bpe_path = out_paths

    paths = [ds_val.src_path, ds_val.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_val.src_bpe_path, ds_val.tgt_bpe_path = out_paths

    paths = [ds_test.src_path, ds_test.tgt_path]
    out_paths = encode(paths, std_ds_train.spm_model_prefix)
    ds_test.src_bpe_path, ds_test.tgt_bpe_path = out_paths

    # dict files from standard_spoc()
    srcdict = std_ds_train.preprocess_dir / f'dict.{src}.txt'
    tgtdict = std_ds_train.preprocess_dir / f'dict.{tgt}.txt'

    src = 'src'
    tgt = 'tgt'
    if finetune_classifier:
        model = 'cls_transformer'
        criterion = 'denoise_criterion'
    else:
        model = 'transformer'
        criterion = 'label_smoothed_cross_entropy'
    save_dir = root / f'models/{exp_id}'
    results_path = root / f'results/{std_exp_id}_pi.txt'

    std_cleaned_pred_path = root / f'results/{std_exp_id}_cleaned.txt'
    std_pred_path = root / f'results/{std_exp_id}.txt'
    denoise_pre_save_path = root / f'models/{pre_exp_id}/checkpoint_best.pt'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {str(strip_ext(ds_train.src_bpe_path))} \
            --validpref {str(strip_ext(ds_val.src_bpe_path))} \
            --testpref {str(strip_ext(ds_test.src_bpe_path))} \
            --destdir {str(ds_train.preprocess_dir)} --joined-dictionary \
            --workers 2 --tgtdict {tgtdict}'
    subprocess.run(shlex.split(cmd))

    label_smoothing = 0.2
    max_epoch = 5
    save_interval = 1

    if denoise_pre:
        # label smoothing 0.2 for 25 epochs, then 0.0 for 5 epochs
        max_epoch = 25
        save_interval = 5

    params = "--encoder-layers 5 --decoder-layers 5 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 8 --decoder-attention-heads 8 "

    if not do_eval or not args.eval_only:
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               {params} \
               --label-smoothing {label_smoothing} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay 0.0001 \
               --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-4 \
               --lr 1e-3 --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch {max_epoch} --save-interval {save_interval} --save-dir {save_dir} --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048"
        if use_denoise_pre:
            cmd += f" --restore-file {denoise_pre_save_path} \
                      --reset-optimizer \
                      --reset-lr-scheduler \
                      --reset-dataloader \
                      --reset-meters"
        if finetune_classifier:
            cmd += f' --pi-restore-path {denoise_save_path} '

        subprocess.run(shlex.split(cmd))

        if denoise_pre:
            label_smoothing = 0.0
            save_interval = 1

            max_epoch += 5
            cmd = f"fairseq-train \
                    {ds_train.preprocess_dir} \
                   --source-lang {src} --target-lang {tgt} \
                   --arch {model} --share-all-embeddings \
                   {params} \
                   --label-smoothing {label_smoothing} \
                   --encoder-normalize-before --decoder-normalize-before \
                   --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
                   --weight-decay 0.0001 \
                   --criterion {criterion} \
                   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
                   --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-4 \
                   --lr 1e-3 --min-lr 1e-9 \
                   --max-tokens 4000 \
                   --update-freq 4 \
                   --max-epoch {max_epoch} --save-interval {save_interval} --save-dir {save_dir} --user-dir {USER_DIR} \
                   --max-source-positions 2048 --max-target-positions 2048"
            subprocess.run(shlex.split(cmd))

        else:
            max_epoch += 1
            label_smoothing = 0.1
            cmd = f"fairseq-train \
                    {ds_train.preprocess_dir} \
                   --source-lang {src} --target-lang {tgt} \
                   --arch {model} --share-all-embeddings \
                   {params} \
                   --label-smoothing {label_smoothing} \
                   --encoder-normalize-before --decoder-normalize-before \
                   --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
                   --weight-decay 0.0001 \
                   --criterion {criterion} \
                   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
                   --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-4 \
                   --lr 1e-3 --min-lr 1e-9 \
                   --max-tokens 4000 \
                   --update-freq 4 \
                   --max-epoch {max_epoch} --save-interval {save_interval} --save-dir {save_dir} --user-dir {USER_DIR} \
                   --max-source-positions 2048 --max-target-positions 2048"
            subprocess.run(shlex.split(cmd))

            max_epoch += 3
            label_smoothing = 0.0
            cmd = f"fairseq-train \
                    {ds_train.preprocess_dir} \
                   --source-lang {src} --target-lang {tgt} \
                   --arch {model} --share-all-embeddings \
                   {params} \
                   --label-smoothing {label_smoothing} \
                   --encoder-normalize-before --decoder-normalize-before \
                   --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
                   --weight-decay 0.0001 \
                   --criterion {criterion} \
                   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
                   --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-4 \
                   --lr 1e-3 --min-lr 1e-9 \
                   --max-tokens 4000 \
                   --update-freq 4 \
                   --max-epoch {max_epoch} --save-interval {save_interval} --save-dir {save_dir} --user-dir {USER_DIR} \
                   --max-source-positions 2048 --max-target-positions 2048"
            subprocess.run(shlex.split(cmd))

    del ds_val, ds_test

    if not finetune_classifier and not denoise_pre:
        denoise_save_path = Path(save_dir) / "checkpoint_best.pt"
        eval_dir = ds_train.preprocess_dir / "evaluate_2step_denoise"
        results_path_classifier = Path(results_path).parent / (Path(results_path).stem  + '_classifier.txt')
        denoise_classifier_save_path = root / f'models/{exp_id}_classifier/checkpoint_best.pt'
        encode_fn = lambda paths: encode(paths, std_ds_train.spm_model_prefix)

        if do_eval or args.eval_only:
            evaluate_spoc_round2(std_cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, do_evaluate_full=True, split='testp', exp_id=None)
            evaluate_spoc_round2(std_cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path, do_evaluate_full=True, split='testp', exp_id=exp_id + " (base + denoise)")
            evaluate_spoc_round2(std_cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, do_evaluate_full=True, split='testw', exp_id=None)
            evaluate_spoc_round2(std_cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path, do_evaluate_full=True, split='testw', exp_id=exp_id + " (base + denoise)")


def pretrain(train_all=False):
    exp_id = 'pretrained_spoc'
    denoise_exp_id = 'denoise_spoc'
    std_exp_id = 'direct_spoc'

    root = Path(ROOT)
    ds_train = SPoC(root / 'data', split='train')
    ds_val = SPoC(root / 'data', split='val')
    ds_test = SPoC(root / 'data', split='test')

    src = 'src'
    tgt = 'tgt'
    model = 'transformer'
    criterion = 'label_smoothed_cross_entropy'

    save_dir = root / f'models/{exp_id}'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    results_path = root / f'results/{exp_id}.txt'
    results_path_2 = root / f'results/{exp_id}_pi.txt'

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

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {strip_ext(ds_train.src_path)} \
            --validpref {strip_ext(ds_val.src_path)} \
            --testpref {strip_ext(ds_test.src_path)} \
            --destdir {ds_train.preprocess_dir} \
            --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    warmup_updates = 4000
    warmup_init_lr = 1e-7
    lr = 1e-3
    label_smoothing = 0.2
    max_epoch=100

    if not args.eval_only:
        params = "--encoder-layers 5 --decoder-layers 5 \
               --encoder-embed-dim 256 --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 8 --decoder-attention-heads 8 "
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               {params} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay 0.0001 \
               --label-smoothing 0.2 --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
               --lr {lr} --min-lr 1e-9 \
               --max-tokens 8000 \
               --update-freq 2 \
               --max-epoch {max_epoch} --save-interval 25 \
               --save-dir {save_dir} --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --best-checkpoint-metric loss \
               --label-smoothing {label_smoothing} \
               --validate-interval 5 \
               --restore-file {denoise_save_path} \
               --reset-optimizer \
               --reset-lr-scheduler \
               --reset-dataloader \
               --reset-meters"
        subprocess.run(shlex.split(cmd))

        label_smoothing = 0.1
        max_epoch += 25

        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               {params} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay 0.0001 \
               --label-smoothing 0.2 --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
               --lr {lr} --min-lr 1e-9 \
               --max-tokens 8000 \
               --update-freq 2 \
               --max-epoch {max_epoch} --save-interval 25 \
               --save-dir {save_dir} --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --best-checkpoint-metric loss \
               --label-smoothing {label_smoothing} --validate-interval 5"
        subprocess.run(shlex.split(cmd))

        label_smoothing = 0.05
        max_epoch += 25

        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               {params} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay 0.0001 \
               --label-smoothing 0.2 --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates {warmup_updates} --warmup-init-lr {warmup_init_lr} \
               --lr {lr} --min-lr 1e-9 \
               --max-tokens 8000 \
               --update-freq 2 \
               --max-epoch {max_epoch} --save-interval 25 \
               --save-dir {save_dir} --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --best-checkpoint-metric loss \
               --label-smoothing {label_smoothing} --validate-interval 5"

    # dict files
    srcdict = ds_train.preprocess_dir / f'dict.{src}.txt'
    tgtdict = ds_train.preprocess_dir / f'dict.{tgt}.txt'

    std_cleaned_pred_path = Path(results_path)
    std_cleaned_pred_path = std_cleaned_pred_path.parent / f"{std_cleaned_pred_path.stem}_cleaned.txt"

    denoise_save_path = str(root / f'models/{denoise_exp_id}/checkpoint_best.pt')
    # with pi
    eval_dir = ds_train.preprocess_dir / 'evaluate_2step_pretrained'
    results_path_classifier = Path(results_path).parent / (Path(results_path).stem  + '_pi_classifier.txt')
    denoise_classifier_save_path =  str(root / f'models/{denoise_exp_id}_classifier/checkpoint_best.pt')
    encode_fn = lambda paths: encode(paths, ds_train.spm_model_prefix)

    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testp', exp_id=exp_id)
    evaluate_spoc_round2(std_cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, do_evaluate_full=True, split='testp', exp_id=None)
    evaluate_spoc_round2(std_cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testp', exp_id=exp_id + " (pretrain + denoise)")

    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testw', exp_id=exp_id)
    evaluate_spoc_round2(std_cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, do_evaluate_full=True, split='testw', exp_id=None)
    evaluate_spoc_round2(std_cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testw', exp_id=exp_id + " (pretrain + denoise)")


def composed(train_all=False, use_f=True):
    root = Path(ROOT)
    ds_train = SPoC(root / 'data', split='train')
    ds_val = SPoC(root / 'data', split='val')
    ds_test = SPoC(root / 'data', split='test')

    exp_id = 'composed_spoc'
    denoise_exp_id = 'denoise_spoc'
    std_exp_id = 'direct_spoc'
    pretrain_exp_id = 'pretrained_spoc'

    if not use_f:
        exp_id += '_nopretraininit'

    src = 'src'
    tgt = 'tgt'
    model = 'double_transformer'
    criterion = 'reinforce_criterion'

    save_dir = root / f'models/{exp_id}'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    standard_save_path = root / f'models/{std_exp_id}/checkpoint_best.pt'
    pretrain_save_path = root / f'models/{pretrain_exp_id}/checkpoint_best.pt'
    results_path = root / f'results/{exp_id}.txt'
    results_path_2 = root / f'results/{exp_id}_pi.txt'

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

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {strip_ext(ds_train.src_path)} \
            --validpref {strip_ext(ds_val.src_path)} \
            --testpref {strip_ext(ds_test.src_path)} \
            --destdir {ds_train.preprocess_dir} \
            --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))

    if not args.eval_only:
        params = "--encoder-layers 5 --decoder-layers 5 \
               --encoder-embed-dim 256 --decoder-embed-dim 256 \
               --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
               --encoder-attention-heads 8 --decoder-attention-heads 8 "
        params2 = "--encoder-layers-2 5 --decoder-layers-2 5 \
               --encoder-embed-dim-2 256 --decoder-embed-dim-2 256 \
               --encoder-ffn-embed-dim-2 1024 --decoder-ffn-embed-dim-2 1024 \
               --encoder-attention-heads-2 8 --decoder-attention-heads-2 8 "
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
               --source-lang {src} --target-lang {tgt} \
               --arch {model} --share-all-embeddings \
               {params} {params2} \
               --encoder-normalize-before --decoder-normalize-before \
               --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
               --weight-decay 0.0001 \
               --label-smoothing 0.2 --criterion {criterion} \
               --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
               --lr-scheduler inverse_sqrt --warmup-updates 2000 --warmup-init-lr 1e-7 \
               --lr 1e-4 --min-lr 1e-9 \
               --max-tokens 4000 \
               --update-freq 4 \
               --max-epoch 20 --save-interval 1 \
               --save-dir {save_dir} --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --pi-restore-path {denoise_save_path} \
               --ce-loss-lambda 1.0 \
               --best-checkpoint-metric loss_2 \
               --keep-best-checkpoints 4 \
               --label-smoothing 0.05 \
               --sampling"
        if not use_f:
            cmd += f' --f-restore-path {pretrain_save_path} '
        else:
            cmd += f' --f-restore-path {standard_save_path} '
        subprocess.run(shlex.split(cmd))

    # dict files
    srcdict = ds_train.preprocess_dir / f'dict.{src}.txt'
    tgtdict = ds_train.preprocess_dir / f'dict.{tgt}.txt'

    cleaned_pred_path = Path(results_path)
    cleaned_pred_path = cleaned_pred_path.parent / f"{cleaned_pred_path.stem}_cleaned.txt"

    encode_fn = lambda paths: encode(paths, ds_train.spm_model_prefix)

    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testp', exp_id=exp_id + " (base predictor)")
    # with pi
    eval_dir = ds_train.preprocess_dir / 'evaluate_2step_composed'
    results_path_classifier = Path(results_path).parent / (Path(results_path).stem  + '_pi_classifier.txt')
    denoise_classifier_save_path =  root / f'models/{denoise_exp_id}_classifier/checkpoint_best.pt'
    evaluate_spoc_round2(cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, split='testp', exp_id=None)
    evaluate_spoc_round2(cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testp', exp_id=exp_id)


    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testw', exp_id=exp_id + " (base_predictor)")
    evaluate_spoc_round2(cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, split='testw', exp_id=None)
    evaluate_spoc_round2(cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testw', exp_id=exp_id)


def backtranslation_spoc(composed=False):
    root = Path(ROOT)
    ds_train = SPoC(root / 'data', split='train')
    ds_val = SPoC(root / 'data', split='val')
    ds_test = SPoC(root / 'data', split='test')

    denoise_ds_train = SPoC(root / 'data', split='train', denoise=True)

    exp_id = 'backtranslation_spoc'

    # train and encode spm
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
    criterion = 'label_smoothed_cross_entropy'
    save_dir = root / f'models/{exp_id}_reverse'
    results_path = root / f'results/{exp_id}_reverse.txt'
    preprocess_dir = Path(str(ds_train.preprocess_dir) + '_reverse')
    preprocess_dir.mkdir(exist_ok=True)
    srcdict = ds_train.preprocess_dir / f'dict.{src}.txt'
    if not srcdict.exists():
        srcdict = None

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {strip_ext(ds_train.src_bpe_path)} \
            --validpref {strip_ext(ds_val.src_bpe_path)} \
            --testpref {strip_ext(ds_test.src_bpe_path)} \
            --destdir {preprocess_dir} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))
    params = "--encoder-layers 5 --decoder-layers 5 \
           --encoder-embed-dim 256 --decoder-embed-dim 256 \
           --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
           --encoder-attention-heads 8 --decoder-attention-heads 8 "
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-7 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 100 --save-interval 25 --save-dir {save_dir} \
           --label-smoothing 0.2 --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-7 \
           --lr 2e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 200 --save-interval 25 --save-dir {save_dir} \
           --label-smoothing 0.1 --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))

    # encode unlabeled data and generate pseudo-inputs

    paths = [denoise_ds_train.src_path, denoise_ds_train.tgt_path]
    out_paths = encode(paths, ds_train.spm_model_prefix)
    denoise_ds_train.src_bpe_path, denoise_ds_train.tgt_bpe_path = out_paths
    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {strip_ext(denoise_ds_train.src_bpe_path)} \
            --destdir {denoise_ds_train.preprocess_dir} \
            --joined-dictionary --workers 2 --srcdict {srcdict}'
    subprocess.run(shlex.split(cmd))
    cmd = f"fairseq-generate {denoise_ds_train.preprocess_dir} \
        --source-lang {src} --target-lang {tgt} \
        --gen-subset train \
        --path {save_dir / 'checkpoint_best.pt'} \
        --beam 1 --remove-bpe sentencepiece --user-dir {USER_DIR}"
    with open(results_path, 'w') as f:
        subprocess.run(shlex.split(cmd), stdout=f)

    src = 'src'
    tgt = 'tgt'
    reverse_save_dir = save_dir
    save_dir = root / f'models/{exp_id}'
    reverse_results_path = results_path
    results_path = root / f'results/{exp_id}.txt'
    preprocess_dir = Path(str(ds_train.preprocess_dir) + '_forward')
    skip_generation = False
    if not preprocess_dir.exists():
        shutil.rmtree(str(preprocess_dir))
        preprocess_dir.mkdir(exist_ok=False)
    else:
        skip_generation = True
        print("Forward dataset already exists - skipping generation")

    # encode and preprocess the forward data
    pseudo_inputs_file = Path(parse_pred_file(reverse_results_path))
    pseudo_outputs_file = Path(denoise_ds_train.tgt_path)
    labeled_inputs_file = Path(ds_train.src_path)
    labeled_outputs_file = Path(ds_train.tgt_path)

    src_path = preprocess_dir / labeled_inputs_file.name
    tgt_path = preprocess_dir / labeled_outputs_file.name

    if not skip_generation:
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
    # train with pseudo labels
    paths = [src_path, tgt_path]
    encoded_paths = encode(paths, ds_train.spm_model_prefix)

    cmd = f'fairseq-preprocess --source-lang {src} --target-lang {tgt} \
            --trainpref {strip_ext(ds_train.src_bpe_path)} \
            --validpref {strip_ext(ds_val.src_bpe_path)} \
            --testpref {strip_ext(ds_test.src_bpe_path)} \
            --destdir {preprocess_dir} \
            --joined-dictionary --workers 2'
    subprocess.run(shlex.split(cmd))
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 20000 --warmup-init-lr 1e-7 \
           --lr 1e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 10 --save-interval 5 --save-dir {save_dir} \
           --label-smoothing 0.2 --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))
    cmd = f"fairseq-train \
            {preprocess_dir} \
           --source-lang {src} --target-lang {tgt} \
           --arch {model} --share-all-embeddings \
           {params} \
           --encoder-normalize-before --decoder-normalize-before \
           --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
           --weight-decay 0.0001 \
           --criterion {criterion} \
           --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
           --lr-scheduler inverse_sqrt --warmup-updates 20000 --warmup-init-lr 1e-7 \
           --lr 1e-3 --min-lr 1e-9 \
           --max-tokens 4000 \
           --update-freq 4 \
           --max-epoch 15 --save-interval 5 --save-dir {save_dir} \
           --label-smoothing 0.1 --user-dir {USER_DIR} \
           --max-source-positions 2048 --max-target-positions 2048"
    subprocess.run(shlex.split(cmd))
    encode_fn = lambda paths: encode(paths, ds_train.spm_model_prefix)

    # finetune on labeled data
    if composed:
        vanilla_bt_save_dir = root / f'models/{exp_id}_finetune'
        exp_id += '_composed'
        model = 'double_transformer'
        criterion = 'reinforce_criterion'

    bt_checkpoint = save_dir / 'checkpoint_best.pt'
    save_dir = root / f'models/{exp_id}_finetune'
    results_path = root / f'results/{exp_id}_finetune.txt'
    results_path_2 = root / f'results/{exp_id}_finetune_pi.txt'
    if composed:
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
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
               --max-epoch 100 --save-interval 5 --save-dir {save_dir} \
               --label-smoothing 0.2 --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --pi-restore-path {reverse_save_dir / 'checkpoint_best.pt'} \
               --ce-loss-lambda 1.0 --best-checkpoint-metric loss_2 \
               --sampling --f-restore-path {vanilla_bt_save_dir / 'checkpoint_best.pt'}"
        subprocess.run(shlex.split(cmd))
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
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
               --max-epoch 150 --save-interval 5 --save-dir {save_dir} \
               --label-smoothing 0.1 --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --pi-restore-path {reverse_save_dir / 'checkpoint_best.pt'} \
               --ce-loss-lambda 1.0 --best-checkpoint-metric loss_2 \
               --sampling --f-restore-path {vanilla_bt_save_dir / 'checkpoint_best.pt'}"
        subprocess.run(shlex.split(cmd))
    else:
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
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
               --max-epoch 100 --save-interval 5 --save-dir {save_dir} \
               --label-smoothing 0.2 --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --restore-file {bt_checkpoint} \
               --reset-optimizer \
               --reset-lr-scheduler \
               --reset-dataloader \
               --reset-meters "
        subprocess.run(shlex.split(cmd))
        cmd = f"fairseq-train \
                {ds_train.preprocess_dir} \
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
               --max-epoch 150 --save-interval 5 --save-dir {save_dir} \
               --label-smoothing 0.1 --user-dir {USER_DIR} \
               --max-source-positions 2048 --max-target-positions 2048 \
               --restore-file {bt_checkpoint} \
               --reset-optimizer \
               --reset-lr-scheduler \
               --reset-dataloader \
               --reset-meters "
        subprocess.run(shlex.split(cmd))

    # dict files
    if composed:
        exp_id_before_denoise = exp_id + " (base predictor)"
        exp_id_after_denoise = exp_id
    else:
        exp_id_before_denoise = exp_id
        exp_id_after_denoise = exp_id + " (bt + denoise)"

    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testp', exp_id=exp_id_before_denoise)
    # with pi
    eval_dir = ds_train.preprocess_dir / 'evaluate_2step_bt'
    if composed:
        eval_dir += '_composed'
    denoise_exp_id = 'denoise_spoc'
    denoise_save_path = root / f'models/{denoise_exp_id}/checkpoint_best.pt'
    results_path_classifier = Path(results_path).parent / (Path(results_path).stem  + '_pi_classifier.txt')
    denoise_classifier_save_path =  root / f'models/{denoise_exp_id}_classifier/checkpoint_best.pt'
    cleaned_pred_path = Path(results_path)
    cleaned_pred_path = cleaned_pred_path.parent / f"{cleaned_pred_path.stem}_cleaned.txt"

    evaluate_spoc_round2(cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, split='testp', exp_id=None)
    evaluate_spoc_round2(cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testp', exp_id=exp_id_after_denoise)

    evaluate_spoc(f'{save_dir}/checkpoint_best.pt', results_path, encode_fn, srcdict, root, split='testw', exp_id=exp_id_before_denoise)
    # with pi
    evaluate_spoc_round2(cleaned_pred_path, denoise_classifier_save_path, eval_dir, encode_fn, srcdict, root, results_path_classifier, split='testw', exp_id=None)
    evaluate_spoc_round2(cleaned_pred_path, denoise_save_path, eval_dir, encode_fn, srcdict, root, results_path_2, split='testw', exp_id=exp_id_after_denoise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scripts')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='only evaluate')
    args = parser.parse_args()

    stats = []

    direct()
    denoise(use_denoise_pre=True)
    denoise(do_eval=False)
    denoise(finetune_classifier=True)
    denoise(do_eval=True)
    pretrain()
    composed(use_f=True)
    composed(use_f=False)
    backtranslation_spoc(composed=False)
    backtranslation_spoc(composed=True)

    res = pd.DataFrame(stats)
    res = res.round(4)
    res.to_csv('spoc_results.tsv', sep='\t', index=None)
    print(res)
