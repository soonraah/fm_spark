package org.apache.spark.ml.fm

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
    val paramMap = new ParamGridBuilder()
      .addGrid(fm.regParam, Array(0.1, 0.01))
      .addGrid(fm.maxIter, Array(5))
      .addGrid(fm.minLabel, Array(1.0))
      .addGrid(fm.maxLabel, Array(5.0))
      .build()

    val cvModel = new CrossValidator()
      .setEstimator(fm)
      .setEstimatorParamMaps(paramMap)
      .setEvaluator(new RegressionEvaluator())
      .setNumFolds(2)
      .fit(dfTraining)

    // Test
    cvModel.transform(dfEvaluation).show(100)
  }

  private def createRatingDataFrame(spark: SparkSession, ratingFile: String): DataFrame = {
    // userId,movieId,rating,timestamp
    val udfCrateFeatureVec = udf {
      (userId: Int, movieRatings: Seq[_], currentMovie: Int) => {
        val featureMap = movieRatings
          .map { s =>
            val items = s.toString.split(":")
            (items(0).toInt, items(1).toDouble)   // other movies rated
          }
          .filter(_._1 != currentMovie)
          .map { t => (t._1 + MaxUserId + MaxMovieId, t._2)}
          .toMap + (userId -> 1.0) + (MaxUserId + currentMovie -> 1.0)

        Vectors.sparse(MaxUserId + MaxMovieId + MaxMovieId, featureMap.toSeq)
      }
    }

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
        collect_list(col("movieRating")) as "movieRatings"
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
