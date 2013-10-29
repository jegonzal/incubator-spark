package org.mlbase.ml

import breeze.linalg._
import spark.RDD

case class BPMeansModelParameters(
  trainingTime: Array[Long],
  features: Array[DenseVector[Double]],
  numFeatures: Array[Int],
  numProposed: Array[Int],
  numAccepted: Array[Int],
  objVals: Array[Double]
)// extends ModelParameters(trainingTime.last)

case class BPMeansModel (
  data: RDD[DenseVector[Double]],
  trainParams: BPMeansParameters,
  params: BPMeansModelParameters
)// extends Model(data, trainParams, params) 
{

  def predict(x: DenseVector[Double]) : Null = {
    null
  }

  def explain() = {
    // TODO
  }

}




