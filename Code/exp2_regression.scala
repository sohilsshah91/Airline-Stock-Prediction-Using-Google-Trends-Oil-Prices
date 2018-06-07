import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

val df_ret=sqlContext.read.parquet("hdfs:///user/ssp573/Reg_All_Data.parquet")
val df1 = df_ret.select(df_ret("Adjusted_Close").as("label"),$"Open_2",$"Change_Percent")
/*df1.columns
for (w<-df1.columns){
println(w+s" "+df1.stat.corr("label",w))
}*/
df1.show(10)
val assemb = new VectorAssembler().setInputCols(Array("Open_2","Change_Percent")).setOutputCol("features") 
val df2 = assemb.transform(df1).select($"label",$"features")
val lr=new LinearRegression().setSolver("l-bfgs")
val lim=(df2.count()*3/4).toInt
val train_df=df2.limit(lim)
val test_df=df2.except(train_df)
train_df.show()
test_df.show()

println(s"_________Linear Regression_________")
println(s"_____Training_____")
val lrModel = lr.fit(train_df)
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
val trainingSummary = lrModel.summary
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")


println(s"_____Testing_____")
val predictions= lrModel.transform(test_df)
predictions.select("prediction","label","features").show()
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


println()
println(s"_________Isotonic Regression_________")
import org.apache.spark.ml.regression.IsotonicRegression

// Trains an isotonic regression model.
val ir = new IsotonicRegression()
val model = ir.fit(train_df)

println(s"Boundaries in increasing order: ${model.boundaries}\n")
println(s"Predictions associated with the boundaries: ${model.predictions}\n")
println(s"Coefficients: ${ir.coefficients} Intercept: ${ir.intercept}")

// Makes predictions.
val predictions_Iso=model.transform(test_df)
predictions_Iso.select("prediction","label","features").show()
val rmse = evaluator.evaluate(predictions_Iso)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)


println(s"_________Decision Tree Regression_________")
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df2)
val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3))
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
val model = pipeline.fit(trainingData)
val predictions_DT = model.transform(testData)
predictions_DT.select("prediction","label","features").show()
val rmse = evaluator.evaluate(predictions_DT)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)



