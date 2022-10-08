import os

max_lens = [128, 256]
augment_ops = ['drop_col', 'sample_row', 'sample_row_ordered', 'shuffle_col', 'drop_cell', 'drop_num_col', 'drop_nan_col', 'shuffle_row']
sampling_methods = ['head', 'random', 'constant', 'frequent', 'tfidf_token', 'tfidf_entity']

for ml in max_lens:
    for ao in [augment_ops[4]]:
        for sm in sampling_methods:
            for run_id in range(5):
              # add --single_column for baseline
                cmd = """python run_pretrain.py \
                  --task %s \
                  --batch_size 64 \
                  --lr 5e-5 \
                  --lm roberta \
                  --n_epochs 3 \
                  --max_len %d \
                  --size 10000 \
                  --save_model \
                  --single_column \
                  --augment_op %s \
                  --fp16 \
                  --sample_meth %s \
                  --run_id %d""" % ("small", ml, ao, sm, run_id)
                print(cmd)
                os.system('sbatch -c 1 -G 1 -J my-exp --tasks-per-node=1 --output=slurm.out --wrap="%s"' % cmd)