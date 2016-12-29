package org.apache.spark.ml.fm

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.HasStepSize
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

private[fm] trait FactorizationMachinesSGDParams
  extends FactorizationMachinesParams
  with HasStepSize
{
  val miniBatchFraction = new Param[Double](this, "miniBatchFraction", "Minibatch flaction [0, 1] for each iterations")

  def getMiniBatchFraction: Double = $(miniBatchFraction)
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

  override def fit(dataset: Dataset[_]): FactorizationMachinesModel = {
    transformSchema(dataset.schema)

    val sparkSession = dataset.sparkSession

    // create empty model
    val initialModel = new FactorizationMachinesModel(
      uid,
      getDimFactorization,
      0.0,
      sparkSession.createDataset(Seq[Strength]()),
      sparkSession.createDataset(Seq[FactorizedInteraction]())
    )

    val dsSampleIndexed = FactorizationMachinesModel
      .addSampleId(dataset, $(sampleIdCol))

    runMiniBatchSGD(dsSampleIndexed, getRegParam, initialModel)
  }

  private def runMiniBatchSGD(data: DataFrame,
                              regParam: Double,
                              initialModel: FactorizationMachinesModel): FactorizationMachinesModel = {
    import data.sparkSession.implicits._

    val dfData = data.cache()

    def udfVecToMap = udf { (vec: Vector) => {
      val m = scala.collection.mutable.Map[Int, Double]()
      vec.foreachActive { (i, value) => m += (i -> value) }
      m
    } }

    val dfMiniBatchArray = dfData
      .randomSplit(Array.fill(getMaxIter)(getMiniBatchFraction), 1234L)

    dfMiniBatchArray
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
          // Explode features
          val dfData = dfMiniBatch
            .select(
              $"sampleId",
              $"label",
              explode(udfVecToMap(col("features"))).as(Seq("featureId", "featureValue"))
            )

          // label, sampleId, featureId, prediction, loss, deltaWi, deltaVi
          val dfLossGrad = model.calcLossGrad(dfData).cache()

          // Calculate loss
          val lossSum = dfLossGrad
            .groupBy(col("sampleId"))
            .agg(first(col("loss")) as "loss")
            .map(_.getAs[Double]("loss"))
            .reduce(_ + _)
          log.info(s"Loss of Iteration $iter: $lossSum")

          // Update weights
          val dfUpdated = dfLossGrad
            .select(
              ((col($(predictionCol)) - col($(labelCol))) * col("deltaWi")) as "deltaWi",
              ((col($(predictionCol)) - col($(labelCol))) * col("deltaVi")) as "deltaVi"
            )
            .groupBy(col("featureId"))
            .agg(
              ((sum("deltaWi") / miniBatchSize) * currentStepSize) as "deltaWiSum",
              ((sum("deltaVi") / miniBatchSize) * currentStepSize) as "deltaViSum"
            )
            .join(model.dimensionStrength, col("featureId") === model.dimensionStrength("id"), "outer")
            .join(model.factorizedInteraction, col("featureId") === model.factorizedInteraction("id"), "outer")
            .select(
              col("featureId"),
              (col("strength") - col("deltaWiSum")) as "strength",
              FactorizationMachinesModel.udfVecMinusVec(col("vec"), col("deltaViSum")) as "factorizedInteraction"
            )
            .select(    // L1 regularization
              col("featureId"),
              (signum(col("strength")) * greatest(lit(0.0), abs(col("strength")) - shrinkageVal)) as "strength",
              (signum(col("factorizedInteraction")) * greatest(lit(0.0), abs(col("factorizedInteraction")) - shrinkageVal)) as "factorizedInteraction"
            )
            .cache()

          // Create a new model
          new FactorizationMachinesModel(
            uid,
            model.dimFactorization,
            model.globalBias,
            dfUpdated
              .map {
                row => Strength(row.getAs[Long]("featureId"), row.getAs[Double]("strength"))
              },
            dfUpdated
              .map {
                row => FactorizedInteraction(row.getAs[Long]("featureId"), row.getAs[DenseVector]("factorizedInteraction"))
              }
          )
        }
      }
    }
  }

  override def copy(extra: ParamMap): Estimator[FactorizationMachinesModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}