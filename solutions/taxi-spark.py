import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

spark = SparkSession.builder \
        .appName("Test SPARK") \
        .master("spark://svmass2.mass.uhb.fr:7077") \
        .config('spark.hadoop.parquet.enable.summary-metadata', 'true') \
        .getOrCreate()


spark.sparkContext.setLogLevel("ERROR")

print("""
*******************************************************
*******************************************************
*******************************************************
*******************************************************
*******************************************************
""")


#df = spark.read.csv("hdfs://svmass2.mass.uhb.fr:54310/user/datasets/nyc-tlc/2014/yellow_tripdata_2014-12.csv", header="true",inferSchema="true")

#df.write.parquet("hdfs://svmass2.mass.uhb.fr:54310/user/navaro_p/2014-12.parquet")

columns = ['pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'payment_type', 'fare_amount', 'tip_amount', 'total_amount']

df = (spark.read.parquet('hdfs://svmass2.mass.uhb.fr:54310/user/navaro_p/nyc-taxi/2012.parquet').select(*columns))

#print(df.head())

#print(" Total ammount of passengers ")
#result = df.agg({'passenger_count': 'sum'}).collect()
#print(result)
#print(" Average number of passenger per trip ")
#print(df.agg({'passenger_count': 'avg'}).collect())
print("How many trip with 0,1,2,3,...,9 passenger")

result = df.groupby('passenger_count').agg({'*': 'count'}).collect()

for i, x  in enumerate(result):
    
    print( f" item {i}  =  \t {x.asDict()} ")
    
passenger = pd.DataFrame([x.asDict() for x in result])

passenger.plot(kind='bar')
plt.savefig("passenger.png")



counts = passenger.sort_values(by=['passenger_count'])

def hour(date):
    return date.hour

from pyspark.sql.functions import hour

result = (df.filter(df.fare_amount > 0)
     .withColumn("hour", hour(df.pickup_datetime))
    .groupby('hour').agg({"tip_amount":'sum'})).collect()

for i, x  in enumerate(result):
    
    print( f"{i} = \t {x.asDict()} ")

spark.stop()

