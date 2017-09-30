package org.apache.spark.ml.fm

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.types.{FloatType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  *
  */
private[fm] trait FactorizationMachinesModelParams
  extends Params
    with HasFeaturesCol
    with HasPredictionCol
    with HasLabelCol
{
  val sampleIdCol = new Param[String](this, "sampleIdCol", "column name for sample ID")
}

/**
  *
  * @param uid
  * @param dimFactorization
  * @param globalBias
  * @param dimensionStrength
  * @param factorizedInteraction
  */
class FactorizationMachinesModel(override val uid: String,
                                 val dimFactorization: Int,
                                 val globalBias: Double,
                                 val dimensionStrength: Dataset[Strength],
                                 val factorizedInteraction: Dataset[FactorizedInteraction])
  extends Model[FactorizationMachinesModel] with FactorizationMachinesModelParams {

  override def copy(extra: ParamMap): FactorizationMachinesModel = {
    val copied = new FactorizationMachinesModel(uid, dimFactorization, globalBias, dimensionStrength, factorizedInteraction)
    copyValues(copied, extra).setParent(parent)
  }

  @org.apache.spark.annotation.Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val dfSampleIndexed = FactorizationMachinesModel.addSampleId(dataset, $(sampleIdCol)).cache()

    val predicted = predict(dfSampleIndexed)

    dfSampleIndexed
      .join(predicted, dfSampleIndexed("sampleId") === predicted("sampleId"), "left_outer")
      .drop(dfSampleIndexed("sampleId"))
      .drop(predicted("sampleId"))
      .na.fill(globalBias, Seq($(predictionCol)))
  }


  private def predict(dfSampleIndexed: DataFrame): DataFrame = {
    val bcW0 = dfSampleIndexed.sqlContext.sparkContext.broadcast(globalBias)

    dfSampleIndexed
      .select(
        dfSampleIndexed("sampleId"),
        explode(FactorizationMachinesModel.udfVecToMap(dfSampleIndexed($(featuresCol)))) as Seq("featureId", "featureValue")
      )
      .join(dimensionStrength, col("featureId") === dimensionStrength("id"), "inner")
      .join(factorizedInteraction, col("featureId") === factorizedInteraction("id"), "inner")
      .select(
        col("sampleId"),
        dimensionStrength("strength") * col("featureValue") as "wixi",
        FactorizationMachinesModel.udfVecMultipleByScalar(factorizedInteraction("vec"), col("featureValue")) as "vfxi",
        FactorizationMachinesModel.vi2xi2(factorizedInteraction("vec"), col("featureValue")) as "vi2xi2"
      )
      .groupBy(col("sampleId"))
      .agg(
        sum(col("wixi")) as "wixiSum",
        (new VectorSum(dimFactorization))(col("vfxi")) as "vfxiSum",
        sum(col("vi2xi2")) as "vi2xi2Sum"
      )
      .select(
        col("sampleId"),
        (FactorizationMachinesModel.sumVx(col("vfxiSum"), col("vi2xi2Sum")) + col("wixiSum") + bcW0.value) as $(predictionCol)
      )
  }

  def calcLossGrad(dfSampleIndexed: DataFrame): DataFrame = {
    val bcW0 = dfSampleIndexed.sqlContext.sparkContext.broadcast(globalBias)

    dfSampleIndexed
      .select(
        col($(labelCol)),
        col("sampleId"),
        explode(FactorizationMachinesModel.udfVecToMap(col($(featuresCol)))) as Seq("featureId", "featureValue")
      )
      .join(dimensionStrength, col("featureId") === dimensionStrength("id"), "left_outer")
      .join(factorizedInteraction, col("featureId") === factorizedInteraction("id"), "left_outer")
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        coalesce(col("strength"), lit(0.0)),
        coalesce(col("vec"), lit(Vectors.zeros(dimFactorization))) as "factorizedInteraction",
        col("featureValue") as "xi",
        dimensionStrength("strength") * col("featureValue") as "wixi",
        FactorizationMachinesModel.udfVecMultipleByScalar(factorizedInteraction("vec"), col("featureValue")) as "vfxi",
        FactorizationMachinesModel.vi2xi2(factorizedInteraction("vec"), col("featureValue")) as "vi2xi2"
      )
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        col("xi"),
        col("wixi"),
        col("vfxi"),
        col("vi2xi2"),
        FactorizationMachinesModel.udfVecMultipleByScalar(col("vfxi"), col("xi")) as "vfxi2",
        (new VectorSum(dimFactorization))(col("vfxi")).over(Window.partitionBy(col("sampleId"))) as "vfxiSum"
      )
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        col("wixi"),
        col("vfxi"),
        col("vi2xi2"),
        col("xi") as "deltaWi",
        FactorizationMachinesModel.udfVecMinusVec(
          FactorizationMachinesModel.udfVecMultipleByScalar(col("vfxiSum"), col("xi")),
          col("vfxi2")
        ) as "deltaVi",
        col("vfxiSum")
      )
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        sum(col("wixi")).over(Window.partitionBy(col("sampleId"))) as "wixiSum",
        sum(col("vi2xi2")).over(Window.partitionBy(col("sampleId"))) as "vi2xi2Sum",
        col("vfxiSum"),
        col("deltaWi"),
        col("deltaVi")
      )
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        (FactorizationMachinesModel.sumVx(col("vfxiSum"), col("vi2xi2Sum")) + col("wixiSum") + bcW0.value) as $(predictionCol),
        col("deltaWi"),
        col("deltaVi")
      )
      .select(
        col($(labelCol)),
        col("sampleId"),
        col("featureId"),
        col($(predictionCol)),
        pow(col($(predictionCol)) - col($(labelCol)), 2.0) as "loss",
        col("deltaWi"),
        col("deltaVi")
      )
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(predictionCol), FloatType)
  }
}

object FactorizationMachinesModel {
  val udfVecToMap: UserDefinedFunction = udf {
    (vec: Vector) => {
      val m = scala.collection.mutable.Map[Int, Double]()
      vec.foreachActive { (i, value) => m += (i -> value) }
      m
    }
  }

  val udfVecMultipleByScalar: UserDefinedFunction = udf {
    (v: Vector, d: Double) => Vectors.fromBreeze(v.asBreeze * d)
  }

  val vi2xi2: UserDefinedFunction = udf {
    (vi: Vector, xi: Double) => vi.toArray.map { vif => vif * vif }.sum * xi * xi
  }

  val sumVx: UserDefinedFunction = udf {
    (vfxiSum: Vector, vi2xi2Sum: Double) => 0.5 * (vfxiSum.toArray.map { vf => vf * vf }.sum - vi2xi2Sum)
  }

  val udfVecMinusVec: UserDefinedFunction = udf {
    (v1: Vector, v2: Vector) => Vectors.fromBreeze(v1.asBreeze - v2.asBreeze)
  }

  def addSampleId(dataset: Dataset[_], columnName: String): DataFrame = dataset
    .select(
      dataset("*"),
      monotonically_increasing_id() as columnName
    )
}

/**
  * The strength of the i-th feature
  *
  * @param id feature ID
  * @param strength strength (w_i)
  */
case class Strength(id: Long, strength: Double)

/**
  * Factorized interaction between i-th and j-th features
  *
  * @param id feature ID
  * @param vec factorized interaction as vector (v_i) with length k
  */
case class FactorizedInteraction(id: Long, vec: DenseVector)
