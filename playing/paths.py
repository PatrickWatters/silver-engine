import os
path = '/Users/patrickwatters/Projects/datasets/sdl-cifar10/train/'

print(os.path.basename(path.removesuffix('/')))


for dirpath, dirnames, filenames in os.walk(path):
    print(dirpath)
    print(filenames)


