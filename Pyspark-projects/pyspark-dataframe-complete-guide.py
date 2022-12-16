# I am working on this
# https://github.com/hyunjoonbok/PySpark/blob/master/PySpark%20Dataframe%20Complete%20Guide%20(with%20COVID-19%20Dataset).ipynb

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
spark = SparkSession.builder.appName("Testing").getOrCreate()

cases = spark.read.load("/Users/kamalsapkota/Desktop/big-data-development/data/Case.csv",
                        format="csv",
                        sep=",",
                        inferSchema="true",
                        header="true")

# First few rows in the file
cases.show()

# this is converting the dataframe into pandas as it is easier to play with
print(cases.limit(10).toPandas())

# to change the column name

cases = cases.withColumnRenamed("infection_case","infection_source")
print(cases.show())

# to change all the columns
cases = cases.toDF(*['case_id', 'province', 'city', 'group', 'infection_case', 'confirmed',
       'latitude', 'longitude'])
print(cases.show())