import argparse
import json
# import run
import torch.cuda


def get_parser():
    parser = argparse.ArgumentParser()
    items = {}
    with open('args.json') as f:
        items = json.load(f)
    parser.add_argument('-LR', "--lr", default=items['lr'], type=float)
    parser.add_argument('-E', "--epochs", default=items['epochs'], type=int)
    parser.add_argument('-S', "--step", default=items['step'], type=int)
    parser.add_argument('-BS', "--batch_size", default=items['batch_size'], type=int)
    parser.add_argument('-OP', "--optim", default=items['optim'], type=str)
    parser.add_argument('-T', "--is_train", action='store_true' if not items['is_train'] else 'store_false')
    parser.add_argument('-CUDA', '--cuda', action='store_true' if not items['cuda'] else 'store_false')
    parser.add_argument('-D', '--device', default=items['device'] if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_known_args()[0]
    # debug
    # print(args)
    return args


if __name__ == '__main__':
    args = get_parser()
    print(args.device)
    # run(args)
