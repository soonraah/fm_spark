package org.apache.spark.ml.fm

import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{IntParam, ParamValidators}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

import scala.collection.mutable


private[fm] trait FactorizationMachinesParams
  extends FactorizationMachinesModelParams
    with HasMaxIter
    with HasLabelCol
    with HasFitIntercept
    with HasRegParam
{
  /**
    * Param for dimensionality of the matrix factorization (positive).
    * Default: 10
    * @group param
    */
  val dimFactorization = new IntParam(this, "dimFactorization", "dimensionality of the factorization", ParamValidators.gtEq(1))

  /** @group getParam */
  def getDimFactorization: Int = $(dimFactorization)

  setDefault(dimFactorization -> 10)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    schema
  }
}

/**
  * Aggregate function to sum vector column
  *
  * @param vecSize vector size
  */
class VectorSum(vecSize: Int) extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = new StructType().add("vector", new VectorUDT())

  override def bufferSchema: StructType = new StructType().add("buf", ArrayType(DoubleType))

  override def dataType: DataType = new VectorUDT()

  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = buffer.update(0, Array.fill(vecSize)(0.0))

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (input.isNullAt(0)) return

    val buf = buffer.getAs[mutable.WrappedArray[Double]](0)
    val inputVector = input.getAs[Vector](0)

    for (i <- buf.indices) {
      buf(i) += inputVector(i)
    }

    buffer.update(0, buf)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val buf1 = buffer1.getAs[mutable.WrappedArray[Double]](0)
    val buf2 = buffer2.getAs[mutable.WrappedArray[Double]](0)

    for ((value, i) <- buf2.zipWithIndex) {
      buf1(i) += value
    }

    buffer1.update(0, buf1)
  }

  override def evaluate(buffer: Row): Any = Vectors.dense(buffer.getAs[mutable.WrappedArray[Double]](0).toArray)
}
