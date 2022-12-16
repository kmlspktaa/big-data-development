# I am working on this:
# this is the reference I am taking from
# https://github.com/hyunjoonbok/PySpark/blob/master/PySpark%20Dataframe%20Complete%20Guide%20(with%20COVID-19%20Dataset).ipynb

import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time

import pyspark  # only run this after findspark.init()
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

cases = cases.withColumnRenamed("infection_case", "infection_source")
print(cases.show())

# to change all the columns
cases = cases.toDF(*['case_id', 'province', 'city', 'group', 'infection_case', 'confirmed',
                     'latitude', 'longitude'])
print(cases.show())

# select a subset of columns
cases = cases.select('province', 'city', 'infection_case', 'confirmed')
print(cases.show())

# ascending sort the column
print(cases.sort("confirmed").show())

# Descending Sort
from pyspark.sql import functions as F

cases.sort(F.desc("confirmed")).show()

# change the column type
from pyspark.sql.types import DoubleType, IntegerType, StringType

cases = cases.withColumn('confirmed', F.col('confirmed').cast(IntegerType()))
cases = cases.withColumn('city', F.col('city').cast(StringType()))

cases.show()

# filter conditions
cases.filter((cases.confirmed > 10) & (cases.province == 'Daegu')).show()

# groupby
from pyspark.sql import functions as F

cases.groupBy(["province", "city"]).agg(F.sum("confirmed").alias("sum_of_confirmed"),
                                        F.max("confirmed").alias("max_of_confirmed")).show()

# rename the column name with alias

cases.groupBy(["province","city"]).agg(
    F.sum("confirmed").alias("TotalConfirmed"),\
    F.max("confirmed").alias("MaxFromOneConfirmedCase")\
    ).show()


# joins

regions = spark.read.load("/Users/kamalsapkota/Desktop/big-data-development/data/Region.csv",
                          format="csv",
                          sep=",",
                          inferSchema="true",
                          header="true")

print(regions.limit(10).toPandas())


# Left Join 'Case' with 'Region' on Province and City column
cases = cases.join(regions, ['province','city'],how='left')
print(cases.limit(10).toPandas())


# use sql with dataframes
cases.registerTempTable('cases_table')
newDF = spark.sql("select * from cases_table where confirmed > 100")
newDF.show()

# new column and add up

import pyspark.sql.functions as F

casesWithNewConfirmed = cases.withColumn("NewConfirmed", 100 + F.col("confirmed"))
casesWithNewConfirmed.show()

# We can also use math functions like F.exp function:
casesWithExpConfirmed = cases.withColumn("ExpConfirmed", F.exp("confirmed"))
casesWithExpConfirmed.show()

# Pyspark UDFs

import pyspark.sql.functions as F
from pyspark.sql.types import *


def casesHighLow(confirmed):
    if confirmed < 50:
        return 'low'
    else:
        return 'high'


# convert to a UDF Function by passing in the function and return type of function
casesHighLowUDF = F.udf(casesHighLow, StringType())
CasesWithHighLow = cases.withColumn("HighLow", casesHighLowUDF("confirmed"))
CasesWithHighLow.show()

cases.printSchema()

#
# Using Pandas UDF
# This allows you to use pandas functionality with Spark. I generally use it when I have to run a groupBy operation on a Spark dataframe or whenever I need to create rolling features
#
# The way we use it is by using the F.pandas_udf decorator. We assume here that the input to the function will be a pandas data frame
#
# The only complexity here is that we have to provide a schema for the output Dataframe. We can use the original schema of a dataframe to create the outSchema.
#

from pyspark.sql.types import IntegerType, StringType, DoubleType, BooleanType
from pyspark.sql.types import StructType, StructField

# Declare the schema for the output of our function

# outSchema = StructType([StructField('case_id',IntegerType(),True),
#                         StructField('province',StringType(),True),
#                         StructField('city',StringType(),True),
#                         StructField('group',BooleanType(),True),
#                         StructField('infection_case',StringType(),True),
#                         StructField('confirmed',IntegerType(),True),
#                         StructField('latitude',StringType(),True),
#                         StructField('longitude',StringType(),True),
#                         StructField('normalized_confirmed',DoubleType(),True)
#                        ])
# decorate our function with pandas_udf decorator
# @F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
# def subtract_mean(pdf):
#     # pdf is a pandas.DataFrame
#     v = pdf.confirmed
#     v = v - v.mean()
#     pdf['normalized_confirmed'] = v
#     return pdf
#
# confirmed_groupwise_normalization = cases.groupby("infection_case").apply(subtract_mean)
#
# confirmed_groupwise_normalization.limit(10).toPandas()

print("Working from here")
# Spark window Functions
# We will simply look at some of the most important and useful window functions available.


timeprovince = spark.read.load("/Users/kamalsapkota/Desktop/big-data-development/data/TimeProvince.csv",
                          format="csv",
                          sep=",",
                          inferSchema="true",
                          header="true")

timeprovince.show()


# ranking
from pyspark.sql.window import Window
windowSpec = Window().partitionBy(['province']).orderBy(F.desc('confirmed'))
cases.withColumn("rank",F.rank().over(windowSpec)).show()

#
# Lag Variables
# Sometimes our data science models may need lag based features. For example, a model might have variables like the price last week or sales quantity the previous day. We can create such features using the lag function with window functions. \
#
# Here I am trying to get the confirmed cases 7 days before. I am filtering to show the results as the first few days of corona cases were zeros. You can see here that the lag_7 day feature is shifted by 7 days.

from pyspark.sql.window import Window

windowSpec = Window().partitionBy(['province']).orderBy('date')

timeprovinceWithLag = timeprovince.withColumn("lag_7",F.lag("confirmed", 7).over(windowSpec))

timeprovinceWithLag.filter(timeprovinceWithLag.date>'2020-03-10').show()


# Rolling Aggregations
# For example, we might want to have a rolling 7-day sales sum/mean as a feature for our sales regression model. Let us calculate the rolling mean of confirmed cases for the last 7 days here. This is what a lot of the people are already doing with this dataset to see the real trends.


from pyspark.sql.window import Window

# we only look at the past 7 days in a particular window including the current_day.
# Here 0 specifies the current_row and -6 specifies the seventh row previous to current_row.
# Remember we count starting from 0.

# If we had used rowsBetween(-7,-1), we would just have looked at past 7 days of data and not the current_day
windowSpec = Window().partitionBy(['province']).orderBy('date').rowsBetween(-6,0)

timeprovinceWithRoll = timeprovince.withColumn("roll_7_confirmed",F.mean("confirmed").over(windowSpec))

timeprovinceWithRoll.filter(timeprovinceWithLag.date>'2020-03-10').show()


# One could also find a use for rowsBetween(Window.unboundedPreceding, Window.currentRow) function, where we take the rows between the first row in a window and the current_row to get running totals. I am calculating cumulative_confirmed here.


from pyspark.sql.window import Window

windowSpec = Window().partitionBy(['province']).orderBy('date').rowsBetween(Window.unboundedPreceding,Window.currentRow)

timeprovinceWithRoll = timeprovince.withColumn("cumulative_confirmed",F.sum("confirmed").over(windowSpec))

timeprovinceWithRoll.filter(timeprovinceWithLag.date>'2020-03-10').show()


# pivot dataframe

# Sometimes we may need to have the dataframe in flat format. This happens frequently in movie data where we may want to show genres as columns instead of rows. We can use pivot to do this. Here I am trying to get one row for each date and getting the province names as columns.



# pivotedTimeprovince = timeprovince.groupBy('date').pivot('province')\
#     .agg(F.sum('confirmed').alias('confirmed') , F.sum('released').alias('released'))
#
# pivotedTimeprovince.limit(10).toPandas()

