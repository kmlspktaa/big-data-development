
# https://github.com/lucasfusinato/pyspark-stock-exchange-analysis/blob/master/notebooks/quiz_local.ipynb

import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time

import pyspark # only run this after findspark.init()
from pyspark.sql import SparkSession, SQLContext
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

# create a Spark session
spark = SparkSession.builder.appName("Stock-Market Predictions").getOrCreate()

cola_df = spark.read.load("/Users/kamalsapkota/Desktop/big-data-development/stocks-data/KO.csv",
                        format="csv",
                        sep=",",
                        inferSchema="true",
                        header="true")

# First few rows in the file
cola_df.show()

cola_df.printSchema()

print(cola_df.columns)


# Print out the first 5 columns.
for row in cola_df.head(5):
    print(row, '\n')

# Use describe() to learn about the DataFrame.
print(cola_df.describe().show())

summary = cola_df.describe()

summary.show()
# rounding of the values
summary.select(summary['summary'],format_number(summary['Open'].cast('float'), 2).alias('Open'),
               format_number(summary['High'].cast('float'), 2).alias('High'),
               format_number(summary['Low'].cast('float'), 2).alias('Low'),
               format_number(summary['Close'].cast('float'), 2).alias('Close'),
               format_number(summary['Adj Close'].cast('float'), 2).alias('Adj Close'),
               format_number(summary['Volume'].cast('int'),0).alias('Volume')).show()

