package org.apache.spark.mllib.clustering

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.mllib.util._
import org.apache.spark.broadcast._
import org.mlbase.runtime.MLBaseRDDFunctions._

import breeze.linalg._

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




