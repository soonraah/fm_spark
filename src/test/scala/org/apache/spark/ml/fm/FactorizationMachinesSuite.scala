package org.apache.spark.ml.fm

import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FunSuite, Matchers}


trait SparkFunSuite extends FunSuite with BeforeAndAfter with Matchers {
  var spark: SparkSession = _

  before {
    spark = SparkSession
      .builder()
      .appName("Sample")
      .config("spark.master", "local[4]")
      .getOrCreate()
  }

  after {
    spark.stop
  }
}

class FactorizationMachinesModelSuite extends SparkFunSuite {
  test("") {
    // -- setup --
    val spark = this.spark
    import spark.implicits._

    val numFeatureDimensions = 4
    val dimFactorization = 3
    val globalBias = 5.0

    val input = Seq(
      (100, Vectors.dense(1.0, 2.0, 1.5, -1.0)),                                // dense
      (101, Vectors.sparse(numFeatureDimensions, Seq((0, 0.5), (2, -1.5)))),    // sparse
      (102, Vectors.sparse(numFeatureDimensions + 1, Seq((0, 2.0), (4, 1.5)))), // sparse (with unlearned dim)
      (103, Vectors.sparse(numFeatureDimensions, Seq()))                        // empty
    ).toDF("rowId", "features")

    val dimensionStrength = Seq(
      Strength(0, 0.1),
      Strength(1, 0.2),
      Strength(2, 0.3),
      Strength(3, 0.4)
    ).toDS()

    val factorizedInteraction = Seq(
      FactorizedInteraction(0, Vectors.dense(1.0, 2.0, 3.0).toDense),
      FactorizedInteraction(1, Vectors.dense(3.0, 2.0, 1.0).toDense),
      FactorizedInteraction(2, Vectors.dense(-0.1, -0.1, -0.2).toDense),
      FactorizedInteraction(3, Vectors.dense(-0.5, 0.3, 0.0).toDense)
    ).toDS()

    val sus = new FactorizationMachinesModel("uid", dimFactorization, globalBias, dimensionStrength, factorizedInteraction)

    // -- exercise --
    val actual = sus
      .transform(input)
      .collect()
      .sortBy(_.getAs[Int]("rowId"))

    // -- verify --
    actual should have size 4
    actual(0).getAs[Double]("prediction") should be(23.77 +- 1.0E-8)
    actual(1).getAs[Double]("prediction") should be(5.275 +- 1.0E-8)
    actual(2).getAs[Double]("prediction") should be(5.2 +- 1.0E-8)
    actual(3).getAs[Double]("prediction") should be(5.0 +- 1.0E-8)





  }
}

class VectorSumSuite extends SparkFunSuite {
  test("vector sum") {
    // -- setup --
    val spark = this.spark
    import spark.implicits._

    val df = Seq(
      (1, Vectors.dense(0.01, 0.02, 0.03)),
      (1, Vectors.dense(0.1, 0.2, 0.3).toSparse),
      (1, Vectors.dense(1.0, 2.0, 3.0)),
      (1, Vectors.dense(10.0, 20.0, 30.0).toSparse),
      (1, Vectors.dense(100.0, 200.0, 300.0))
    ).toDF("id", "vec")

    // -- exercise --
    val actual = df
      .groupBy('id)
      .agg((new VectorSum(3))('vec))
      .collect()

    // -- verify --
    actual should have size 1
    actual(0).getInt(0) should be(1)
    actual(0).getAs[DenseVector](1) should be(Vectors.dense(111.11, 222.22, 333.33))
  }
}
