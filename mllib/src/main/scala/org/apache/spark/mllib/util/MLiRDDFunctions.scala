package org.apache.spark.mllib.util

import org.apache.spark.rdd.SplitRDD
import org.apache.spark.rdd.RDD



/**
 * Extra functions on RDDs in MLBase, made available through an implicit conversion.
 * As this functionality is incorporated into Spark, these functions should be removed.
 *
 * To use the implicit conversion, `import MLBaseRDDFunctions._`
 */
class MLiRDDFunctions[T: ClassManifest](self: RDD[T]) {

  /**
   * Return a set of RDDs, each of which corresponds to a disjoint subset
   * of this RDD's partitions.
   *
   * The partitions are divided among the RDDs using a range-partitioning scheme;
   * the last RDD may have fewer partitions if this RDD does not have a multiple
   * of `numRDDs` partitions.  If this is undesirable, you can randomly
   * hash-repartition this RDD prior to performing the  `split()`.
   */
  def split(numRDDs: Int): Array[RDD[T]] = {
    if (self.partitions.length < numRDDs) {
      self.partitions.map(part => new SplitRDD(self, Array(part.index)))
    } else {
      (0 until numRDDs).map { i =>
        val rangeStart = ((i.toLong * self.partitions.length) / numRDDs).toInt
        val rangeEnd = (((i.toLong + 1) * self.partitions.length) / numRDDs).toInt
        new SplitRDD(self, (rangeStart until rangeEnd).toArray)
      }.toArray
    }
  }
}


object MLiRDDFunctions {
  implicit def rddToMLBaseRDDFunctions[T: ClassManifest](rdd: RDD[T]) = new MLiRDDFunctions(rdd)
}
