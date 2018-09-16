df.insert(loc=2,column='day',value=np.ones(len(df)))
df.index = pd.to_datetime(df[['year','month','day']])
df = df['mean temp']
