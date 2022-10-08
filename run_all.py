import os

datasets = ['small', 'large']
lms = ['distilbert', 'roberta']
batch_sizes = [32, 64]

for ds in datasets:
    for lm in lms:
        for batch_size in batch_sizes:
            for run_id in range(5):
                cmd = """python train.py \
                  --task %s \
                  --batch_size %s \
                  --lr 5e-5 \
                  --lm %s \
                  --n_epochs 20 \
                  --max_len 128 \
                  --fp16 \
                  --run_id %d""" % (ds, batch_size, lm, run_id)
                print(cmd)
                os.system('sbatch -c 1 -G 1 -J my-exp --tasks-per-node=1 --wrap="%s"' % cmd)
