import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml._

val df_ret=sqlContext.read.parquet("hdfs:///user/ssp573/Reg_AAL_short.parquet")
val test_data=sc.textFile("hdfs://dumbo/user/ssp573/file_daily_input.csv")
val test_data1=test_data.map(p=>p.split(","))
val test_data2= test_data1.map(p=> (p(0).trim.toInt,p(1).trim.toInt,p(2).trim.toInt,p(3).trim.toInt,p(4).trim.toInt,p(5).trim.toInt,p(6).trim.toInt,p(7).trim.toInt,p(8).trim.toInt,p(9).trim.toInt,p(10).trim.toInt,p(11).trim.toInt,p(12).trim.toInt,p(13).trim.toInt,p(14).trim.toInt,p(15).trim.toInt,p(16).trim.toInt,p(17).trim.toInt,p(18).trim.toInt,p(19).trim.toInt)).toDF()

val df1 = df_ret.select(df_ret("Adjusted_Close").as("label"),$"_2",$"_3",$"_4",$"_5",$"_6",$"_7",$"_8",$"_9",$"_10",$"_11",$"_12",$"_13",$"_14",$"_15",$"_16",$"_17",$"_18",$"_19",$"_20",$"_21")
/*df1.columns
for (w<-df1.columns){
println(w+s" "+df1.stat.corr("label",w))
}*/
df1.show(10)
val assemb = new VectorAssembler().setInputCols(Array("_2","_3","_4","_5","_6","_7","_8","_9","_10","_11","_12","_13","_14","_15","_16","_17","_18","_19","_20","_21")).setOutputCol("features") 
val assemb1 = new VectorAssembler().setInputCols(Array("_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11","_12","_13","_14","_15","_16","_17","_18","_19","_20")).setOutputCol("features")

val df2 = assemb.transform(df1).select($"label",$"features")

println()
println(s"_________Decision Tree Regression_________")
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
val test = assemb1.transform(test_data2).select($"features")
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df2)
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
val model = pipeline.fit(df2)
//model.write.save("hdfs:///user/jm7432/sample-model")
//val loaded_model=PipelineModel.read.load("sample-model")
val predictions_DT = model.transform(test)
predictions_DT.show()
predictions_DT.write.mode("append").parquet("hdfs://dumbo/user/ssp573/Save_Output.parquet")
