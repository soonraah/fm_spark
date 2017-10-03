package org.apache.spark.ml.fm

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}


object FactorizationMachinesSample {
  val MaxUserId: Int = 671
  val MaxMovieId: Int = 164979

  def main(args: Array[String]): Unit = {
    // logger setting
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("org.apache.spark.ml.fm").setLevel(Level.INFO)

    // Create SparkSession
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("FactorizationMachinesSample")
      .getOrCreate()

    // Get from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
    val ratingsFile = "data/ml-latest-small/ratings.csv"

    // Create DataFrames
    val dfFeature = createRatingDataFrame(spark, ratingsFile)

    val dfs = dfFeature.randomSplit(Array(0.9, 0.1))
    val dfTraining = dfs(0)     // for cross validation
    val dfEvaluation = dfs(1)   // for test

    val (minLabel, maxLabel) = getMinMaxLabel(dfFeature)

    // Run cross validation
    val fm = new FactorizationMachinesSGD()
      .setMaxIter(5)
      .setMiniBatchFraction(0.2)
      .setMinLabel(minLabel)
      .setMaxLabel(maxLabel)
      .setInitialSd(0.01)
      .setStepSize(1.0)

    val paramMap = new ParamGridBuilder()
      .addGrid(fm.regParam, Array(1.0e-6, 0.0))
      .build()

    val evaluator = new RegressionEvaluator().setMetricName("mae")

    val cvModel = new CrossValidator()
      .setEstimator(fm)
      .setEstimatorParamMaps(paramMap)
      .setEvaluator(evaluator)
      .setNumFolds(2)
      .fit(dfTraining)

    cvModel.avgMetrics.zipWithIndex.foreach { t => println(s"Cross validation MAE ${t._2}: ${t._1}")}

    // Test
    val dfPredicted = cvModel.transform(dfEvaluation).cache()
    dfPredicted.show(100)

    val mae = evaluator.evaluate(dfPredicted)
    println(s"Test MAE: $mae")

    spark.stop()
  }

  private def createRatingDataFrame(spark: SparkSession, ratingFile: String): DataFrame = {
    val udfCrateFeatureVec = udf {
      (userId: Int, movieRatings: Seq[_], currentMovie: Int) => {
        val ratingMap = if (movieRatings.size < 2) {
          Map[Int, Double]()
        } else {
          val ratingWeight = 1.0 / (movieRatings.size - 1.0)
          movieRatings
            .map { s =>
              val items = s.toString.split(":")
              items(0).toInt    // other movies rated
            }
            .filter(_ != currentMovie)
            .map { movieId => (MaxUserId + MaxMovieId + movieId, ratingWeight) }
            .toMap
        }
        val featureMap = ratingMap + (userId -> 1.0, MaxUserId + currentMovie -> 1.0)

        Vectors.sparse(MaxUserId + MaxMovieId + MaxMovieId, featureMap.toSeq)
      }
    }

    // userId,movieId,rating,timestamp
    val dfRating = spark
      .read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .csv(ratingFile)

    dfRating
      .select(
        col("userId"),
        concat(col("movieId"), lit(":"), col("rating")) as "movieRating"
      )
      .groupBy(col("userId"))
      .agg(
        collect_set(col("movieRating")) as "movieRatings"
      )
      .select(
        col("userId"),
        col("movieRatings"),
        explode(col("movieRatings")) as "movieRating"
      )
      .select(
        col("userId"),
        col("movieRatings"),
        split(col("movieRating"), ":")(0).cast(IntegerType) as "movieId",
        split(col("movieRating"), ":")(1).cast(DoubleType) as "rating"
      )
      .select(
        col("rating") as "label",
        udfCrateFeatureVec(col("userId"), col("movieRatings"), col("movieId")) as "features"
      )
  }

  private def getMinMaxLabel(dfLabeled: DataFrame): (Double, Double) = {
    import dfLabeled.sparkSession.implicits._
    dfLabeled
      .map { row => val label = row.getAs[Double]("label"); (label, label) }
      .reduce { (a, b) => (math.min(a._1, b._1), math.max(a._2, b._2)) }
  }
}
