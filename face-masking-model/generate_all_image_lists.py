import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='Directory containing images', type=str)
parser.add_argument('--type', help='Extension of files to include', type=str, default='.jpg')
parser.add_argument('--n', help='Number of files to include', type=int, default=-1)
parser.add_argument('--save_filename', help='Name of file to save list of all images', type=str, default='all_images.txt')
parser.add_argument('--use_seed', help='1 to use seed in randomization, 0 to use no seed', type=int, default=1)
args = parser.parse_args()

if args.use_seed == 1:
    random.seed(42)

def return_files_of_type(path, file_type):
    all_files = []

    for root, directories, files in os.walk(args.dir, topdown=True):
        if len(directories) > 0:
            continue

        for file in files:
            if file.endswith(file_type):
                all_files.append(os.path.join(root, file))

    return all_files

def main():
    with open(os.path.join(args.dir, args.save_filename), 'w') as f:
        selected_files = return_files_of_type(args.dir, args.type)

        if args.n > -1:
            selected_files = random.sample(selected_files, args.n)

        f.write('\n'.join(selected_files))

if __name__ == '__main__':
    main()
