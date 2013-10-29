package org.apache.spark.mllib.featurization

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




