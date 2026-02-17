import glob

if __name__ == '__main__':

    max_seed = 1

    for seed in range(max_seed+1):
        files = glob.glob(f'data/sparseful/*seed-{seed}*')
        if len(files) != 396:
            raise Exception(f'[Sparseful] Expected 396 files, found {len(files)}')

        files = glob.glob(f'data/baselines/*seed-{seed}*')
        if len(files) != 792:
            raise Exception(f'[Baselines] Expected 792 files, found {len(files)}')

