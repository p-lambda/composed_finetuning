# SPoC Dataset

This dataset is developed at Stanford and is circulated under the CC BY 4.0 license.

## Release log

Current version - v1.1

1.1 - Bug report fixed: few erroneous gold programs (40) now removed from the dataset. All the programs currently present in v1.1 should compile & pass all testcases with GCC 5.5, 2020-01-22
1.0.2 - Added our train data split, minor directory restructuring, 2019-10-22
1.0.1 - License moved from CC BY-SA 4.0 to CC BY 4.0 on request, 2019-06-18
1.0 - First public release with arxiv preprint, 2019-06-11

## Contents

`train` directory contains the training data:
1.  `spoc-train.tsv`: Contains the train data. Ideally, split this further into `train-eval-test` to fine-tune and evaluate your algorithms. Our split of the train data is in `split`.

`test` directory contains the test data:
2. `spoc-testp.tsv`: Hidden test-set to evaluate generalization to new problems. Please read the paper for more details.
3. `spoc-testw.tsv`: Hidden test-set to evaluate generalization to new workers. Please read the paper for more details.

The `testcases` directory contains test cases for every problem id. Please use `*_testcases_public.txt` as public test cases for search and evaluate on the `*_testcases_hidden.txt`. Note that different submissions of the same problem have the same test cases.

## Format

The `.tsv` file has the following header 
`text	code	workerid	probid	subid	line	indent`
1. `text` is the human-authored pseudocode for that line.
2. `code` is the gold code line.
3. `workerid`is the unique-id of worker who annotated this program. Ranges from `00` to `58` inclusive.
4. `probid` is the problem id of this program.
5. `subid` is the submission id of this program.
6. `line` is the line number inside the program.
7. `indent` is the indentation level of the program. This gives information of where control-flow blocks end and hence is useful for reconstructing the full program and inserting braces.

Note:
* Please use GCC 5.5 for all your compile/run evaluations. Some constructs like 'gets' are removed in more recent GCC versions.
* New programs start when the `line` field resets back to 0.
* Simple lines like `int main`, `}`and `return 0;` were not annotated, but these are important in the full reconstruction. Hence we have signal this by empty `text` lines. Please use the gold line code when the `text` field is empty.

## Questions

For any questions / troubles / issues regarding the SPoC dataset, please reach out to Sumith Kulal (sumith@cs.stanford.edu) or Ice Pasupat (ppasupat@cs.stanford.edu). We are more than happy to help you out!
