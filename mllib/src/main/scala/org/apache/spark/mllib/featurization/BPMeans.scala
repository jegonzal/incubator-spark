package org.apache.spark.mllib.featurization

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.broadcast._

import org.apache.spark.mllib.util._
import org.apache.spark.mllib.util.MLiRDDFunctions._


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



case class BPMeansParameters(
  lambda: Double,
  blockSize: Int,
  numProcessors: Int,
  maxIterations: Int
) extends AlgorithmParameters

object BPMeansAlgorithm //extends Algorithm[DenseVector[Double], Null, BPMeansParameters]
{

  def parseParameters(params: Map[String, String]) : BPMeansParameters = {
    val lambda = params.get("lambda").get.toDouble
    val blockSize = params.get("blockSize").get.toInt
    val numProcessors = params.get("numProcessors").get.toInt
    val maxIterations = params.get("maxIterations").get.toInt
    new BPMeansParameters(lambda,blockSize,numProcessors,maxIterations)
  }

  def intersectSortedArrays(a: Array[Int], b: Array[Int]) : Array[Int] = {
    var i=0
    var j=0
    val alen = a.length
    val blen = b.length
    val intersect = ArrayBuffer[Int]()
    while (i < alen && j < blen){
      if (a(i) < b(j)){
        i = i + 1
      }else if (a(i) > b(j)){
        j = j + 1
      }else{
        intersect += a(i)
        i = i + 1
        j = j + 1
      }
    }
    intersect.toArray
  }

  def mapZ01(z: Array[Boolean]) : Array[Array[Int]] = {
    val arr0 = ArrayBuffer[Int]()
    val arr1 = ArrayBuffer[Int]()
    ((0 until z.length) zip z).foreach(iz => {
      if (!iz._2){
        arr0 += iz._1
      }else{
        arr1 += iz._1
      }
      })
    Array(arr0.toArray,arr1.toArray)
  }

  def findSameColumns(sets1: Array[Array[Int]], sets2: Array[Array[Int]]) : Array[Array[Int]] = {
    val sameColumns = ArrayBuffer[Array[Int]]()
    for (i <- 0 until sets1.length){
      for (j <- 0 until sets2.length){
        val s = intersectSortedArrays(sets1(i),sets2(j))
        if (s.length>1) sameColumns += s
      }
    }
    sameColumns.toArray
  }

  def serialAcceptFeatures(proposedFeatures_t: Array[DenseVector[Double]], lambda: Double) : Array[DenseVector[Double]] = {
    //proposedFeatures_t.foreach(println)
    var newFeatures = ArrayBuffer[DenseVector[Double]]()
    var proposedFeatures = proposedFeatures_t
    while (proposedFeatures.length > 0){
      // Pick up first item as feature
      val newF = proposedFeatures(0)
      if (norm(newF) >= lambda){
        //println("New feature: " + newF)
        newFeatures += newF
        // Add new feature
        proposedFeatures = proposedFeatures.map(pf0 => {
          val pf1 = pf0 - newF
          val norm0 = norm(pf0)
          val norm1 = norm(pf1)
          if (norm1 < norm0) pf1 else pf0
          })
      }
      proposedFeatures = proposedFeatures.drop(1)
    }
    return newFeatures.toArray
  }

  def serialAcceptFeatures_old(proposedFeatures_t: Array[DenseVector[Double]], lambda: Double) : Array[DenseVector[Double]] = {
    //proposedFeatures_t.foreach(println)
    var newFeatures = ArrayBuffer[DenseVector[Double]]()
    var proposedFeatures = proposedFeatures_t
    while (proposedFeatures.length > 0){
      // Pick up first item as feature
      val newF = proposedFeatures(0)
      //println("New feature: " + newF)
      newFeatures += newF
      // Add new feature
      val proposedFeatures_norm = proposedFeatures.map(pf0 => {
        val pf1 = pf0 - newF
        val norm0 = norm(pf0)
        val norm1 = norm(pf1)
        if (norm1 < norm0) (pf1,norm1) else (pf0,norm0)
        })
      // Filter out small features
      proposedFeatures = proposedFeatures_norm.filter(pfn => (pfn._2>=lambda)).map(_._1)
    }
    return newFeatures.toArray
  }

  def sampleZ(x: DenseVector[Double], z: Array[Boolean], features_bc: Broadcast[Array[DenseVector[Double]]])
  : (Array[Boolean],DenseVector[Double],Double) = {
    val features = features_bc.value
    var residual = x.copy
    (0 until z.length).foreach(i => {
      if (z(i)) residual = residual - features(i)
      })
    var newZ = new Array[Boolean](features.length)
    (0 until z.length).foreach(i => newZ(i) = z(i))
    var resNorm = residual.norm(2)
    (0 until newZ.length).foreach(i => {
      var res0     = if ( newZ(i)) residual + features(i) else residual.copy
      var res1     = if (!newZ(i)) residual - features(i) else residual.copy
      var resNorm0 = if ( newZ(i)) norm(res0) else resNorm
      var resNorm1 = if (!newZ(i)) norm(res1) else resNorm
      if (resNorm0 < resNorm1){
        residual = res0
        resNorm = resNorm0
        newZ(i) = false
      }else{
        residual = res1
        resNorm = resNorm1
        newZ(i) = true
      }
      })
    return (newZ,residual,resNorm)
  }

  def removeColumns(z: Array[Boolean], delColumns: Array[Int]) : Array[Boolean] = {
    val newZ = ArrayBuffer[Boolean]()
    var j = 0
    val dclen = delColumns.length
    ((0 until z.length) zip z).foreach(iz => {
      val i = iz._1
      var dropCol = if (j < dclen) (if (i==delColumns(j)) true else false) else false
      if (dropCol){
        j = j + 1
      }else{
        newZ += iz._2
      }
      })
    newZ.toArray
  }

  def outerProduct(z: Array[Boolean], x: DenseVector[Double]) : DenseMatrix[Double] = {
    val zlen = z.length
    val xlen = x.length
    val zTx = DenseMatrix.zeros[Double](zlen,xlen)
    ((0 until zlen) zip z).foreach(iz => if (iz._2) zTx(iz._1,::) := x)
    zTx
  }

  def computeObjVal(X: Array[RDD[DenseVector[Double]]], Z: Array[RDD[Array[Boolean]]], features: Array[DenseVector[Double]], lambda: Double, dim: Int, sc: SparkContext) : Double = {
    val features_bc = sc.broadcast(features)
    val objval = features.length * lambda*lambda + (X zip Z).map(XZ => {
      (XZ._1 zip XZ._2).map(xz => {
        val n = norm(xz._1 -(xz._2 zip features_bc.value).map(zf => if (zf._1) zf._2 else zf._2*0.0).foldLeft(DenseVector.zeros[Double](dim))(_+_))
        n*n
        }).reduce(_+_)
      }).reduce(_+_)
    objval
  }

  def train(data: RDD[DenseVector[Double]], params: BPMeansParameters, sc: SparkContext) : BPMeansModel = {

    // Parameters
    val lambda = params.lambda
    val blockSize = params.blockSize
    val numProcessors = params.numProcessors
    val maxIterations = params.maxIterations

    // Statistics
    val startTime = System.currentTimeMillis
    var numIterations = 0
    var numFeatures = ArrayBuffer[Int]()
    var trainTime = ArrayBuffer[Long]()
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

    // Initialize features, Z
    var features = Array[DenseVector[Double]]()
    var Z = X.map(xi => xi.map(_ => Array[Boolean]()))
    objvals += computeObjVal(X,Z,features,lambda,dim,sc)

    // Bootstrap!
    features = serialAcceptFeatures(X(0).take(ceil((blockSize*numProcessors).toDouble/16.0).toInt), lambda)

    var hasConverged = false
    while (!hasConverged && numIterations < maxIterations){
      println("Iteration #" + numIterations)

      val iterStart = System.currentTimeMillis
      val oldFeatures = features
      var features_bc = sc.broadcast(features)
      // Iterate through blocks to sample Z and learn new features
      ((0 until numEpochs) zip (X zip Z)).foreach(tXZ => {
        val t = tXZ._1
        val xt = tXZ._2._1
        val zt = tXZ._2._2
        print("\tepoch " + t + " of " + X.length)
        // Sample Z, and compute residuals
        //val features_bc = sc.broadcast(features)
        val Z_residual = (xt zip zt).map(xz => sampleZ(xz._1,xz._2,features_bc)).cache//.persist(StorageLevel.MEMORY_ONLY_SER)
        // Propose new features, serially accept
        val proposedFeatures = Z_residual.filter(_._3>lambda).map(_._2).collect
        numProposed += proposedFeatures.length
        val newFeatures = serialAcceptFeatures(proposedFeatures, lambda)
        numAccepted += newFeatures.length
        if (newFeatures.length > 0){
          features = features ++ newFeatures
          features_bc = sc.broadcast(features)
        }
        // Add new features to sampled Z
        val newFeatures_bc = sc.broadcast(newFeatures)
        Z(t) = (Z_residual.map(_._1) zip
          (Z_residual.map(_._2) zip Z_residual.map(_=> Array[Boolean]()))
          .map(rz => sampleZ(rz._1,rz._2,newFeatures_bc)).map(_._1)).map(
          zz => zz._1 ++ zz._2
          )
        println(" done")
        })

      // Add 0's to make all Z's same length
      Z = Z.map(_.map(z => z ++ (new Array[Boolean](features.length-z.length))))

      print("\tfind same columns, ")
      // Find columns with same entries
      val sameColumns = Z.map(_.map(mapZ01).reduce(findSameColumns)).reduceLeft(findSameColumns)

      print("delete them, ")
      // Delete same columns (keep only the first of each set of same columns)
      val delColumns  = sameColumns.foldLeft(ArrayBuffer[Int]())((buf,cols) => buf ++ cols.drop(1)).sortWith(_ < _).toArray
      if (delColumns.length > 0)
        Z = Z.map(_.map(z=>removeColumns(z,delColumns)))
      println("done.")

      // Compute ZTZ and ZTX
//      val ZTZ = Z.map(
//        _.map(z=>outerProduct(z,new DenseVector[Double](z.map(zz=>if(zz) 1.0 else 0.0))))
//        .reduce(_+_)).reduceLeft(_+_)


      val ZTZ = Z.map(zRDD =>
        zRDD.aggregate[DenseMatrix[Double]](null)(
          (acc, z) => {
            val zlen = z.length
            val newAcc = if(acc != null) acc else DenseMatrix.zeros[Double](zlen,zlen)
            for(i <- 0 until zlen if z(i) ; j <- 0 until zlen if z(j)) { newAcc(i,j) += 1.0 }
           newAcc
          },
          (acc1, acc2) => if(acc1 == null) acc2 else if( acc2 == null) acc1 else acc1 + acc2
        )
      ).reduceLeft(_+_)


      val ZTX = (X zip Z).map{
        case (xRDD, zRDD) => {
          xRDD.zip(zRDD).aggregate[DenseMatrix[Double]](null)(
            (acc, xz) => {
              val x = xz._1
              val z = xz._2
              val zlen = z.length
              val xlen = x.length
              val newAcc = if(acc != null) acc else DenseMatrix.zeros[Double](zlen,xlen)
              for(i <- 0 until zlen if z(i); j <- 0 until xlen) { newAcc(i,j) += x(j) }
              newAcc
            },
            (acc1, acc2) => if(acc1 == null) acc2 else if( acc2 == null) acc1 else acc1 + acc2
          )
        }
      }.reduceLeft(_+_)

      println("\tcompute stats complete")

      // Compute new features
      val At = (ZTZ \ ZTX).t
      val featuresBuffer = ArrayBuffer[DenseVector[Double]]()
      (0 until At.cols).foreach(i => featuresBuffer += At(::,i))
      features = featuresBuffer.toArray

      val objval = computeObjVal(X,Z,features,lambda,dim,sc)
      println("\tobjval = " + objval)

      // Check for convergence
      numIterations += 1
      if (features.length == oldFeatures.length){
        val sumNorms = (features zip oldFeatures).map(fo => norm(fo._1-fo._2)).reduce(_+_)
        //if (sumNorms==0) hasConverged = true
        println("\tsumNorms = " + sumNorms)
      }else{
        println("\t#features changed from " + oldFeatures.length + " to " + features.length)
      }

      // Collect stats
      numFeatures += features.length
      trainTime += (System.currentTimeMillis - startTime)
      objvals += objval

      println("\ttime = " + (System.currentTimeMillis-iterStart))
    }

    trainTime.toArray.foreach(println(_))

    // Return the Model
    //val trainTime = System.currentTimeMillis - startTime
    val mparams = new BPMeansModelParameters(trainTime.toArray, features, numFeatures.toArray, numProposed.toArray, numAccepted.toArray, objvals.toArray)
    new BPMeansModel(data, params, mparams)

  }
}

