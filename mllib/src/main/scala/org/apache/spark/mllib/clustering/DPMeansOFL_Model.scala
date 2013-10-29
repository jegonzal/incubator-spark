package org.mlbase.ml

import breeze.linalg._
import spark.RDD

case class DPMeansOFLModelParameters(
  trainingTime: Array[Long],
  centers: Array[DenseVector[Double]],
  numFeatures: Array[Int],
  numProposed: Array[Int],
  numAccepted: Array[Int],
  objVals: Array[Double],
  epochTime: Array[Long]
) //extends ModelParameters(trainingTime.last)

case class DPMeansOFLModel (
  data: RDD[DenseVector[Double]],
  trainParams: DPMeansOFLParameters,
  params: DPMeansOFLModelParameters
) //extends Model(data, trainParams, params)
{


}




