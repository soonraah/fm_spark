package org.apache.spark.ml.fm

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.HasStepSize
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.util.Random

private[fm] trait FactorizationMachinesSGDParams
  extends FactorizationMachinesParams
  with HasStepSize
{
  val miniBatchFraction = new Param[Double](this, "miniBatchFraction", "Minibatch flaction [0, 1] for each iterations")

  def getMiniBatchFraction: Double = $(miniBatchFraction)

  val initialSd = new Param[Double](this, "initialSd", "Standard deviation to initialize weights")

  def getInitialSd: Double = $(initialSd)
}


class FactorizationMachinesSGD(override val uid: String)
  extends Estimator[FactorizationMachinesModel]
    with FactorizationMachinesSGDParams {

  def this() = this(Identifiable.randomUID("fm"))

  /** @group setParam */
  def setDimFactorization(value: Int): this.type = set(dimFactorization, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setMiniBatchFraction(value: Double): this.type = set(miniBatchFraction, value)

  def setRegParam(value: Double): this.type = set(regParam, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setMinLabel(value: Double): this.type = set(minLabel, value)

  def setMaxLabel(value: Double): this.type = set(maxLabel, value)

  def setInitialSd(value: Double): this.type = set(initialSd, value)

  setDefault(
    dimFactorization -> 10,
    featuresCol -> "features",
    labelCol -> "label",
    predictionCol -> "prediction",
    sampleIdCol -> "sampleId",
    maxIter -> 10,
    miniBatchFraction -> 0.1,
    regParam -> 0.1,
    stepSize -> 1.0,
    minLabel -> 0.0,
    maxLabel -> 1.0,
    initialSd -> 0.01
  )

  override def fit(dataset: Dataset[_]): FactorizationMachinesModel = {
    transformSchema(dataset.schema)

    // create empty model
    val initialModel = createInitialModel(dataset)

    val dsSampleIndexed = FactorizationMachinesModel
      .addSampleId(dataset, $(sampleIdCol))

    runMiniBatchSGD(dsSampleIndexed, getRegParam, initialModel)
  }

  private def runMiniBatchSGD(data: DataFrame,
                              regParam: Double,
                              initialModel: FactorizationMachinesModel): FactorizationMachinesModel = {
    import data.sparkSession.implicits._

    val dfData = data.cache()

    val udfVecToMap = udf { (vec: Vector) => {
      val m = scala.collection.mutable.Map[Int, Double]()
      vec.foreachActive { (i, value) => m += (i -> value) }
      m
    } }

    val udfL1RegularizationVec = udf {
      (v: Vector, shrinkageVal: Double) => {
        Vectors.dense(
          v.toArray.map { w => math.signum(w) * math.max(0.0, math.abs(w) - shrinkageVal) }
        )
      }
    }

    val udfZeroVector = udf { () => Vectors.zeros(getDimFactorization) }

    val dfMiniBatchArray = dfData
      .randomSplit(Array.fill(getMaxIter)(getMiniBatchFraction), 1234L)

    val trainedModel = dfMiniBatchArray
      .zipWithIndex
      .foldLeft[FactorizationMachinesModel](initialModel) {
      (model, tuple) => {
        val dfMiniBatch = tuple._1
        val iter = tuple._2 + 1

        val currentStepSize = getStepSize / math.sqrt(iter)
        val shrinkageVal = currentStepSize * getRegParam    // for L1 regularization

        val miniBatchSize = dfMiniBatch.count

        if (miniBatchSize == 0) {
          log.warn(s"Iteration ($iter/$getMaxIter). The size of sampled batch is zero")
          model
        } else {
          // label, sampleId, featureId, prediction, loss, deltaWi, deltaVi
          val dfLossGrad = model.calcLossGrad(dfMiniBatch, getInitialSd).cache()

          // Calculate loss
          val lossSum = dfLossGrad
            .groupBy(col("sampleId"))
            .agg(first(col("loss")) as "loss")
            .map(_.getAs[Double]("loss"))
            .reduce(_ + _)
          log.info(s"Loss of Iteration ($iter/$getMaxIter): $lossSum")

          // Update weights
          val dfJoined = dfLossGrad
            .select(
              col("featureId"),
              (col("deltaWi") * col($(predictionCol)) - col($(labelCol))) as "deltaWi",
              FactorizationMachinesModel.udfVecMultipleByScalar(col("deltaVi"), col($(predictionCol)) - col($(labelCol))) as "deltaVi"
            )
            .groupBy(col("featureId"))
            .agg(
              ((sum("deltaWi") / miniBatchSize) * currentStepSize) as "deltaWiSum",
              FactorizationMachinesModel.udfVecMultipleByScalar(
                (new VectorSum(getDimFactorization))(col("deltaVi")),
                lit(currentStepSize / miniBatchSize)
              ) as "deltaViSum"
            )
            .as("grad")
            .join(
              model.dimensionStrength as "ds",
              $"grad.featureId" === $"ds.id",
              "outer"
            )
            .join(
              model.factorizedInteraction as "fi",
              coalesce($"grad.featureId", $"ds.id") === $"fi.id",    // fi が持つ featureId は必ず ds も持つはず
              "outer"
            )

          val dfUpdated = dfJoined
            .select(
              coalesce($"grad.featureId", $"ds.id", $"fi.id") as "featureId",
              (coalesce($"ds.strength", lit(0.0)) - coalesce($"grad.deltaWiSum", lit(0.0))) as "strength",
              FactorizationMachinesModel.udfVecMinusVec(
                coalesce($"fi.vec", udfZeroVector()),
                coalesce($"grad.deltaViSum", udfZeroVector())
              ) as "factorizedInteraction"
            )
            .select(    // L1 regularization
              col("featureId"),
              (signum(col("strength")) * greatest(lit(0.0), abs(col("strength")) - shrinkageVal)) as "strength",
              udfL1RegularizationVec(col("factorizedInteraction"), lit(shrinkageVal)) as "factorizedInteraction"
            )
            .cache()

          // Create a new model
          val updatedModel = new FactorizationMachinesModel(
            uid,
            model.dimFactorization,
            model.globalBias,
            dfUpdated
              .map {
                row => Strength(row.getAs[Int]("featureId"), row.getAs[Double]("strength"))
              }
              .cache(),
            dfUpdated
              .map {
                row => FactorizedInteraction(row.getAs[Int]("featureId"), row.getAs[DenseVector]("factorizedInteraction"))
              }
              .cache()
          )
            .setMinLabel(getMinLabel)
            .setMaxLabel(getMaxLabel)

          // clean
          dfLossGrad.unpersist()
          dfUpdated.unpersist()
          model.dimensionStrength.unpersist()
          model.factorizedInteraction.unpersist()

          updatedModel
        }
      }
    }

    dfData.unpersist()
    trainedModel
  }

  private def createInitialModel(dataset: Dataset[_]): FactorizationMachinesModel = {
    val initialSd = getInitialSd
    val dimFactorization = getDimFactorization

    import dataset.sparkSession.implicits._

    val dsFeatures = dataset
      .select(
        explode(FactorizationMachinesModel.udfVecToMap(col($(featuresCol)))) as Seq("featureId", "featureValue")
      )
      .select(
        col("featureId")
      )
      .distinct()
      .map { row => row.getAs[Int](0) }

    val dsStrength = dsFeatures
      .map(Strength(_, Random.nextGaussian() * initialSd))

    val dsInteraction = dsFeatures
      .map(FactorizedInteraction(
        _,
        Vectors.dense((0 until dimFactorization).map { _ => Random.nextGaussian() * initialSd } .toArray).toDense
      ))

    new FactorizationMachinesModel(
      uid,
      dimFactorization,
      0.0,
      dsStrength,
      dsInteraction
    )
      .setMinLabel(getMinLabel)
      .setMaxLabel(getMaxLabel)
  }

  override def copy(extra: ParamMap): Estimator[FactorizationMachinesModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}