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
