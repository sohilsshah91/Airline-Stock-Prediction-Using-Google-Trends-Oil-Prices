import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

val data = sc.textFile("hdfs:///user/ssp573/daily_adjusted_AAL.csv")
val data_trends= sc.textFile("hdfs:///user/ssp573/file_new.csv")
val header = data.first
val header_trends= data_trends.first
val rows = data.filter(l=>l!=header)
val rows_trends = data_trends.filter(l=>l!=header_trends)
case class CC1(TimeStamp: String, Open: Double, High: Double, Low: Double, Close: Double, Adjusted_Close: Double,Volume: Long)
val allSplit = rows.map(line=>line.split(","))
val allSplit_trends=rows_trends.map(line=>line.split(","))
val allData = allSplit.map(p=> CC1(p(0).toString,p(1).trim.toDouble,p(2).trim.toDouble,p(3).trim.toDouble,p(4).trim.toDouble,p(5).trim.toDouble,p(6).trim.toLong))
val allData_trends= allSplit_trends.map(p=> (p(0).toString,p(1).trim.toInt,p(2).trim.toInt,p(3).trim.toInt,p(4).trim.toInt,p(5).trim.toInt,p(6).trim.toInt,p(7).trim.toInt,p(8).trim.toInt,p(9).trim.toInt,p(10).trim.toInt,p(11).trim.toInt,p(12).trim.toInt,p(13).trim.toInt,p(14).trim.toInt,p(15).trim.toInt,p(16).trim.toInt,p(17).trim.toInt,p(18).trim.toInt,p(19).trim.toInt,p(20).trim.toInt))
val allDF = allData.toDF()
val allDF_trends= allData_trends.toDF()

val df_jomo = allDF.as("dfjomo")
val df_panse = allDF_trends.as("dfpanse")
val joined_df = df_jomo.join(df_panse, col("dfjomo.TimeStamp") === col("dfpanse._1"),"inner")


joined_df.write.parquet("hdfs:///user/ssp573/Reg_AAL_short.parquet")
