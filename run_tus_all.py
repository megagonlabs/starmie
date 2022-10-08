import os
import numpy as np

benchmark = 'wdc'
matching = 'exact'
table_order = 'column'
augment_ops = ['drop_col', 'sample_row', 'sample_row_ordered', 'shuffle_col', 'drop_cell', 'drop_num_col', 'drop_nan_col', 'shuffle_row']
sampling_methods = ['head', 'random', 'constant', 'frequent', 'tfidf_token', 'tfidf_entity', 'tfidf_row']

k = 60
threshold = 0.1
enc = 'cl'
ao = 'drop_col'
sm = 'tfidf_entity'
run_id = 0
cmd = """python test_naive_search.py \
    --encoder %s \
    --benchmark %s \
    --augment_op %s \
    --sample_meth %s \
    --matching %s  \
    --table_order %s \
    --run_id %d \
    --K %d \
    --threshold %f \
    --scal %f""" % (enc, benchmark, ao, sm, matching, table_order, run_id, k, threshold, scale)
                
print(cmd)
os.system('sbatch -c 1 -G 1 -J my-exp --tasks-per-node=1 --output=slurm.out --wrap="%s"' % cmd)
