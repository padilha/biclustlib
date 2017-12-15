import numpy as np

def parse_in_chunks(filename, chunksize, rows_idx, cols_idx):
    with open(filename, 'r') as f:
        all_lines = f.readlines()
        chunk = 4

        for i in range(0, len(all_lines), chunksize):
            rows, cols = all_lines[i + rows_idx], all_lines[i + cols_idx]
            rows = np.array(rows.strip().split(), dtype=np.int)
            cols = np.array(cols.strip().split(), dtype=np.int)
            yield rows, cols
