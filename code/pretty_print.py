import argparse
from collections import defaultdict


def add_newlines(string):
    new_string = ""
    for_started = False
    if_started = False
    num_open_parens = 0
    # replace hashes
    new_string = string.replace('$', '\n')
    new_string = new_string.replace('~', '\t')
    return new_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scripts')
    parser.add_argument('--input', type=str)
    parser.add_argument('--intermediate', type=str, default=None)
    parser.add_argument('--pred', type=str, default=None)
    parser.add_argument('--gold', type=str)
    parser.add_argument('--interactive', action='store_true', default=False)
    parser.add_argument('--diffs', action='store_true', default=False)
    args = parser.parse_args()


    files = {'input': args.input, 'intermediate': args.intermediate, 'pred': args.pred, 'gold': args.gold}

    lists = defaultdict(list)
    for k, fn in files.items():
        if fn is None:
            continue
        with open(fn, 'r') as f:
            for line in f:
                lists[k].append(line)

    for i in range(len(lists['gold'])):
        print_strs = {}
        for k in lists.keys():
            print_str = add_newlines(lists[k][i])
            print_strs[k] = print_str

        if args.diffs:
            if 'pred' in print_strs and 'intermediate' in print_strs:
                if print_strs['pred'] == print_strs['intermediate']:
                    continue

        for k, print_str in print_strs.items():
            print(f"{k.upper()}: \n{print_str}")
            if args.interactive:
                input()
        print('-' * 80)
        print()
        print()
