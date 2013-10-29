package org.mlbase.ml

import scala.util.Random
import spark._
import spark.rdd._
import scala.collection.mutable.ArrayBuffer
import spark.SparkContext
import spark.SparkContext._
import scala.math.ceil
import spark.broadcast._

import org.mlbase.runtime.MLBaseRDDFunctions._

import breeze.linalg._

case class DPMeansOFLParameters(
  lambda: Double,
  blockSize: Int,
  numProcessors: Int,
  maxIterations: Int,
  q: Double
) extends AlgorithmParameters

object DPMeansOFLAlgorithm // extends Algorithm[DenseVector[Double], Null, DPMeansOFLParameters]
{
  Random.setSeed(43)

  def parseParameters(params: Map[String, String]) : DPMeansOFLParameters = {
    val lambda = params.get("lambda").get.toDouble
    val blockSize = params.get("blockSize").get.toInt
    val numProcessors = params.get("numProcessors").get.toInt
    val maxIterations = params.get("maxIterations").get.toInt
    val q = params.get("q").get.toDouble
    new DPMeansOFLParameters(lambda,blockSize,numProcessors,maxIterations,q)
  } 

  def sampleZ(x: DenseVector[Double], centers_bc: Broadcast[Array[DenseVector[Double]]], lambda: Double, q: Double) : (Int,Double) = {
    val centers = centers_bc.value
    val dists = centers.map(c => norm(x-c))
    var kstar_mindist = ((0 until centers.length) zip dists).foldLeft((-1,lambda))((a,b) => if (a._2 < b._2) a else b)
    if (q > 0.0){
      if (Random.nextDouble <= (kstar_mindist._2/lambda)) kstar_mindist = (-1,kstar_mindist._2)
    }
    kstar_mindist
  }

  def serialAcceptCenters(proposedCenters_dist: Array[(DenseVector[Double],Double)], q: Double) : Array[DenseVector[Double]] = {
    //proposedCenters_dist.map(_._1).foreach(println)
    val newCenters = ArrayBuffer[DenseVector[Double]]()
    proposedCenters_dist.foreach(cd => {
      val dists = newCenters.map(nc => norm(nc-cd._1))
      val mindist = dists.foldLeft(cd._2)((a,b) => if (a < b) a else b)
      if (q > 0.0){
        if (Random.nextDouble <= mindist/cd._2) newCenters += cd._1
      }else{
        if (mindist >= cd._2) newCenters += cd._1
      }
      })
    //println("Accepted:")
    //newCenters.foreach(println)
    newCenters.toArray
  }

  def resampleZ(xz: (DenseVector[Double],Int), newCenters_bc: Broadcast[Array[DenseVector[Double]]]) : Int = {
    val newCenters = newCenters_bc.value
    if (xz._2 == -1){
      val dists = newCenters.map(c => norm(xz._1-c))
      ((0 until newCenters.length) zip dists).reduceLeft((a,b) => if (a._2 < b._2) a else b)._1
    }else{
      xz._2
    }
  }

  def computeObjVal(X: Array[RDD[DenseVector[Double]]], Z: Array[RDD[Int]], centers: Array[DenseVector[Double]], lambda: Double, dim: Int, sc: SparkContext) : Double = {
    val centers_bc = sc.broadcast(centers)
    centers.length * lambda*lambda + (X zip Z).map(XZ => {
      (XZ._1 zip XZ._2).map(xz => {
        val n = if (xz._2 == -1) Double.PositiveInfinity else norm(xz._1 - centers_bc.value(xz._2))
        n*n
        }).reduce(_+_)
      }).reduce(_+_)
  }

  def train(data: RDD[DenseVector[Double]], params: DPMeansOFLParameters, sc: SparkContext) : DPMeansOFLModel = {

    // Parameters
    val lambda = params.lambda
    val blockSize = params.blockSize
    val numProcessors = params.numProcessors
    val maxIterations = params.maxIterations
    val q = params.q

    // Statistics
    val startTime = System.currentTimeMillis
    var numIterations = 0
    var numCenters = ArrayBuffer[Int]()
    var trainTime = ArrayBuffer[Long]()
    var epochTime = ArrayBuffer[Long]()
    var numProposed = ArrayBuffer[Int]()
    var numAccepted = ArrayBuffer[Int]()
    var objvals = ArrayBuffer[Double]()

    val N = data.count
    val numBlocks : Int = math.ceil(N.toDouble / blockSize.toDouble).toInt
    val numEpochs : Int = math.ceil(numBlocks.toDouble / numProcessors.toDouble).toInt

    println("Processing " + N + " data points")
    println("\t#blocks = " + numBlocks)
    println("\t#epochs = " + numEpochs)

    // Ignore the labels
    val X = data.split(numEpochs)
    X.map(xi => xi.cache)

    val dim = X(0).first.length

    // Initialize centers, Z
    var centers = Array[DenseVector[Double]]()
    var Z = X.map(xi => xi.map(_=> -1))
    objvals += computeObjVal(X,Z,centers,lambda,dim,sc)

    // Bootstrap!
    if (q < 0.0)
      centers = serialAcceptCenters(X(0).take(ceil((blockSize*numProcessors).toDouble/16.0).toInt).map(x => (x,lambda)), q)

    var hasConverged = false
    while (!hasConverged && numIterations < maxIterations){
      println("Iteration #" + numIterations)

      val iterStart = System.currentTimeMillis
      val oldCenters = centers.map(_.copy)
      var centers_bc = sc.broadcast(centers)
      // Iterate through blocks to sample Z and learn new centers
      (0 until numEpochs).foreach(t => {
        print("\tepoch " + t + " of " + X.length)
        val epochStart = System.currentTimeMillis
        val xt = X(t)
        // Sample Z and compute distances
        val Zdist = xt.map(x => sampleZ(x,centers_bc,lambda,q)).cache
        // Propose new centers, serially accept
        val proposedCenters_dist = (X(t) zip Zdist).filter(_._2._1 == -1).map(xzd => (xzd._1,xzd._2._2)).collect
        numProposed += proposedCenters_dist.length
        val newCenters = serialAcceptCenters(proposedCenters_dist, q)
        numAccepted += newCenters.length
        if (newCenters.length > 0){
          centers = centers ++ newCenters
          centers_bc = sc.broadcast(centers)
        }
        // If doing DP-means (not OFL), update Z
        if (q < 0.0){
          Z(t) = (X(t) zip Zdist.map(_._1)).map(xz => resampleZ(xz, centers_bc))
        }
        val epochEnd = System.currentTimeMillis
        println(" done in " + (epochEnd - epochStart) + "ms")
        epochTime += (epochEnd - epochStart)
        })

      // Compute new centers
      if (q < 0.0){
        val clusterCount = new Array[Int](centers.size)
        val clusterSums  = clusterCount.map(_=> DenseVector.zeros[Double](dim))
        val allStats = (X zip Z).map(XZ => (XZ._1 zip XZ._2).map(xz => (xz._2,(xz._1,1))).reduceByKey{case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}.collect)
        allStats.foreach(_.foreach(zxc => {
            val z = zxc._1
            val x = zxc._2._1
            val c = zxc._2._2
            clusterCount(z) = clusterCount(z) + c
            clusterSums(z)  = clusterSums(z)  + x
          }))
        //((0 until clusterCount.length) zip clusterCount).foreach(println)
        centers = (clusterSums zip clusterCount).map(sc => sc._1 / sc._2.toDouble)
      }

      val objval = computeObjVal(X,Z,centers,lambda,dim,sc)
      println("\tobjval = " + objval)

      // Check for convergence
      numIterations += 1
      if (oldCenters.length == centers.length){
        val sumNorms = (oldCenters zip centers).map(oc => norm(oc._1-oc._2)).reduce(_+_)
        //if (sumNorms==0) hasConverged = true
        println("\tsumNorms = " + sumNorms)
      }else{
        println("\t#centers changed from " + oldCenters.length + " to " + centers.length)
      }

      // Collect stats
      numCenters += centers.length
      trainTime += (System.currentTimeMillis - startTime)
      objvals += objval

      println("\ttime = " + (System.currentTimeMillis-iterStart))
    }

    // Return the Model
    //val trainTime = System.currentTimeMillis - startTime
    val mparams = new DPMeansOFLModelParameters(trainTime.toArray, centers, numCenters.toArray, numProposed.toArray, numAccepted.toArray, objvals.toArray, epochTime.toArray)
    new DPMeansOFLModel(data, params, mparams)

  }
}

