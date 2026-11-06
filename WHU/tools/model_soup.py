"""Equal-weight model soup for UAD checkpoints.

Usage:
  python tools/model_soup.py --inputs logs/A/transformer_150.pth \
      logs/A/transformer_160.pth logs/A/transformer_170.pth \
      --output logs/A/transformer_soup.pth
"""
import argparse

import torch


def load_state(path):
    try:
        obj = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        raise RuntimeError('checkpoint is not a state dict: {}'.format(path))
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']
    if isinstance(obj, dict) and 'model' in obj:
        obj = obj['model']
    return {k.replace('module.', ''): v for k, v in obj.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    states = [load_state(p) for p in args.inputs]
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise RuntimeError('checkpoint keys differ')
    soup = {}
    for key in keys:
        tensors = [s[key] for s in states]
        if tensors[0].dtype.is_floating_point:
            acc = tensors[0].clone().float()
            for t in tensors[1:]:
                acc += t.float()
            acc /= len(tensors)
            soup[key] = acc.to(tensors[0].dtype)
        else:
            soup[key] = tensors[0].clone()
    torch.save(soup, args.output)
    print('saved soup of {} checkpoints to {}'.format(len(states), args.output))


if __name__ == '__main__':
    main()
