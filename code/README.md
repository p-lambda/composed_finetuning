# Pseudocode-to-code

The directory contains the code for pseudocode-to-code experiments, including
a synthetic pseudocode-to-code dataset with 1000 paired examples and 20000 unlabeled examples (and other versions can be generated using the given code),
as well as a processed version of the [SPoC](https://github.com/Sumith1896/spoc) dataset with crowdsourced pseudocode on code from programming competitions on codeforces.com.

The synthetic pseudocode-to-code dataset supplies the model with all but the declaration types, which must be inferred from context. The denoiser helps with global type inference and instantiation decisions, simplifying the task for the base predictor.

We consider *full program* pseudocode-to-code translation, instead of line-by-line as in
previous works. We do not utilize the compiler in any way except for during evaluation,
and thus do not use compiler messages as side information. We also use greedy decoding,
meaning that a beam search could be used (at the expense of added computation) to improve the results.

All the code-based data is in C++.

## Setup

For SPoC unlabeled data, run the script `download_spoc_unlabeled_data.sh`. It should add three new directories to `data/spoc-data/train/split/`, where each directory begins with `preprocess_denoise_*`.

Please install fairseq by going into the `fairseq` directory and running `pip install -e .` in your virtualenv.

In general, if you see file exists errors for dictionary files, this is OK;
the preprocessing to create the dictionaries will just be skipped if they
already exist.

## Run
The synthetic experiment can be run using the command `bash run_all_synthetic.sh`.
The SPoC experiment can be run using the command `python run_spoc.py`.

## Data
The synthetic data is in `data/synthetic_1000a/`, which includes train, val, test splits.
Each split contains 4 data files: 
- `{split}.src` is the pseudocode
- `{split}.tgt` is the code
- `{split}.inp` is the inputs file for the test case,
- `{split}.out` is the outputs file for the correct output of the test case as outputted by the gold code in the `{split}.tgt` file.

Running with different settings in the synthetic dataset will cause the code to generate new data.

The SPoC training data contains SPoC training examples with less than 1000 characters.
The SPoC unlabeled data contains corrupted code according to the corruption method in [this paper](https://arxiv.org/abs/2005.10636).
