df[~df.Cancelled].groupby('Origin').Origin.count().compute()

df.groupby('Origin').DepDelay.mean().compute()

df.groupby('DayOfWeek').DepDelay.mean().compute()
