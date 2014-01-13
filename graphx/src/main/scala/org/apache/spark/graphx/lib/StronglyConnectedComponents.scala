package org.apache.spark.graphx.lib

import scala.reflect.ClassTag

import org.apache.spark.graphx._

object StronglyConnectedComponents {

  /**
   * Compute the strongly connected component (SCC) of each vertex and return a graph with the
   * vertex value containing the lowest vertex id in the SCC containing that vertex.
   *
   * @tparam VD the vertex attribute type (discarded in the computation)
   * @tparam ED the edge attribute type (preserved in the computation)
   *
   * @param graph the graph for which to compute the SCC
   *
   * @return a graph with vertex attributes containing the smallest vertex id in each SCC
   */
  def run[VD: ClassTag, ED: ClassTag](graph: Graph[VD, ED], numIter: Int): Graph[VertexID, ED] = {

    // the graph we update with final SCC ids, and the graph we return at the end
    var sccGraph = graph.mapVertices { case (vid, _) => vid }
    // graph we are going to work with in our iterations
    var sccWorkGraph = graph.mapVertices { case (vid, _) => (vid, false) }.cache()

    var numVertices = sccWorkGraph.numVertices
    var iter = 0
    while (sccWorkGraph.numVertices > 0 && iter < numIter) {
      iter += 1
      do {
        numVertices = sccWorkGraph.numVertices
        sccWorkGraph = sccWorkGraph.outerJoinVertices(sccWorkGraph.outDegrees) {
          (vid, data, degreeOpt) => if (degreeOpt.isDefined) data else (vid, true)
        }.outerJoinVertices(sccWorkGraph.inDegrees) {
          (vid, data, degreeOpt) => if (degreeOpt.isDefined) data else (vid, true)
        }.cache()

        // get all vertices to be removed
        val finalVertices = sccWorkGraph.vertices
            .filter { case (vid, (scc, isFinal)) => isFinal}
            .mapValues { (vid, data) => data._1}

        // write values to sccGraph
        sccGraph = sccGraph.outerJoinVertices(finalVertices) {
          (vid, scc, opt) => opt.getOrElse(scc)
        }
        // only keep vertices that are not final
        sccWorkGraph = sccWorkGraph.subgraph(vpred = (vid, data) => !data._2).cache()
      } while (sccWorkGraph.numVertices < numVertices)

      sccWorkGraph = sccWorkGraph.mapVertices{ case (vid, (color, isFinal)) => (vid, isFinal) }

      // collect min of all my neighbor's scc values, update if it's smaller than mine
      // then notify any neighbors with scc values larger than mine
      sccWorkGraph = Pregel[(VertexID, Boolean), ED, VertexID](sccWorkGraph, Long.MaxValue)(
        (vid, myScc, neighborScc) => (math.min(myScc._1, neighborScc), myScc._2),
        e => {
          if (e.srcId < e.dstId) {
            Iterator((e.dstId, e.srcAttr._1))
          } else {
            Iterator()
          }
        },
        (vid1, vid2) => math.min(vid1, vid2))

      // start at root of SCCs. Traverse values in reverse, notify all my neighbors
      // do not propagate if colors do not match!
      sccWorkGraph = Pregel[(VertexID, Boolean), ED, Boolean](
        sccWorkGraph, false, activeDirection = EdgeDirection.In)(
        // vertex is final if it is the root of a color
        // or it has the same color as a neighbor that is final
        (vid, myScc, existsSameColorFinalNeighbor) => {
          val isColorRoot = vid == myScc._1
          (myScc._1, myScc._2 || isColorRoot || existsSameColorFinalNeighbor)
        },
        // activate neighbor if they are not final, you are, and you have the same color
        e => {
          val sameColor = e.dstAttr._1 == e.srcAttr._1
          val onlyDstIsFinal = e.dstAttr._2 && !e.srcAttr._2
          if (sameColor && onlyDstIsFinal) {
            Iterator((e.srcId, e.dstAttr._2))
          } else {
            Iterator()
          }
        },
        (final1, final2) => final1 || final2)
    }
    sccGraph
  }

}