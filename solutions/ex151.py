
import dask

early = df[df.DepDelay < 0].size
late = df[df.DepDelay > 0].size

early_res, late_res = dask.compute(early, late)
len(df[~df.Cancelled])
