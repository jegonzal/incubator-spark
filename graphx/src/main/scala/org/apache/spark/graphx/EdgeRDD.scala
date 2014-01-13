package org.apache.spark.graphx

import scala.reflect.{classTag, ClassTag}

import org.apache.spark.{OneToOneDependency, Partition, Partitioner, TaskContext}
import org.apache.spark.graphx.impl.EdgePartition
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * `EdgeRDD[ED]` extends `RDD[Edge[ED]]` by storing the edges in columnar format on each partition
 * for performance. It is constructed using [[impl.EdgePartitionBuilder]].
 */
class EdgeRDD[@specialized ED: ClassTag](
    val partitionsRDD: RDD[(PartitionID, EdgePartition[ED])])
  extends RDD[Edge[ED]](partitionsRDD.context, List(new OneToOneDependency(partitionsRDD))) {

  partitionsRDD.setName("EdgeRDD")

  override protected def getPartitions: Array[Partition] = partitionsRDD.partitions

  /**
   * If `partitionsRDD` already has a partitioner, use it. Otherwise assume that the
   * [[PartitionID]]s in `partitionsRDD` correspond to the actual partitions and create a new
   * partitioner that allows co-partitioning with `partitionsRDD`.
   */
  override val partitioner =
    partitionsRDD.partitioner.orElse(Some(Partitioner.defaultPartitioner(partitionsRDD)))

  override def compute(part: Partition, context: TaskContext): Iterator[Edge[ED]] = {
    firstParent[(PartitionID, EdgePartition[ED])].iterator(part, context).next._2.iterator
  }

  override def collect(): Array[Edge[ED]] = this.map(_.copy()).collect()

  override def persist(newLevel: StorageLevel): EdgeRDD[ED] = {
    partitionsRDD.persist(newLevel)
    this
  }

  /** Persist this RDD with the default storage level (`MEMORY_ONLY`). */
  override def persist(): EdgeRDD[ED] = persist(StorageLevel.MEMORY_ONLY)

  /** Persist this RDD with the default storage level (`MEMORY_ONLY`). */
  override def cache(): EdgeRDD[ED] = persist()

  override def unpersist(blocking: Boolean = true): EdgeRDD[ED] = {
    partitionsRDD.unpersist(blocking)
    this
  }

  def mapEdgePartitions[ED2: ClassTag](f: (PartitionID, EdgePartition[ED]) => EdgePartition[ED2])
    : EdgeRDD[ED2] = {
    new EdgeRDD[ED2](partitionsRDD.mapPartitions({ iter =>
      val (pid, ep) = iter.next()
      Iterator(Tuple2(pid, f(pid, ep)))
    }, preservesPartitioning = true))
  }

  def innerJoin[ED2: ClassTag, ED3: ClassTag]
      (other: EdgeRDD[ED2])
      (f: (VertexID, VertexID, ED, ED2) => ED3): EdgeRDD[ED3] = {
    val ed2Tag = classTag[ED2]
    val ed3Tag = classTag[ED3]
    new EdgeRDD[ED3](partitionsRDD.zipPartitions(other.partitionsRDD, true) {
      (thisIter, otherIter) =>
        val (pid, thisEPart) = thisIter.next()
        val (_, otherEPart) = otherIter.next()
        Iterator(Tuple2(pid, thisEPart.innerJoin(otherEPart)(f)(ed2Tag, ed3Tag)))
    })
  }

  def collectVertexIDs(): RDD[VertexID] = {
    partitionsRDD.flatMap { case (_, p) => Array.concat(p.srcIds, p.dstIds) }
  }
}