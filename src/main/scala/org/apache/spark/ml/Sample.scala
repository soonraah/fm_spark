package org.apache.spark.ml

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Sample {
  def main(args: Array[String]): Unit = {
    println("--- START ---")

    val spark = SparkSession
      .builder()
      .appName("Sample")
//      .config("spark.some.config.option", "some-value")
      .config("spark.master", "local[4]")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val dataset = spark.read.format("libsvm").load("data/sample.txt")
    dataset.printSchema()

    println("----")

    dataset.show()

    def vec2map = udf { (vec: Vector) => {
      val m = scala.collection.mutable.Map[Int, Double]()
      vec.foreachActive { (i, value) => m += (i -> value) }
      m
    } }

    def breezize = udf { (vec: Vector) => vec.asBreeze.asInstanceOf[Vector] }

    dataset
      .select(
        '*,
        breezize('features) as "br",
        explode(vec2map('features)) as Seq("kkk", "vvv"),
        monotonically_increasing_id() as "id"
      )
      // .groupBy('label)
      // .select('label, breezize('features))
      .show()

    spark.stop()

    println("---- END ----")
  }
}
