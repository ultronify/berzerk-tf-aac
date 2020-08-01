import argparse
from train import train
from data_cleaning import run_data_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train(render=args.render)
    elif args.mode == 'explore':
        run_data_experiment()
