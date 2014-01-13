---
layout: global
title: GraphX Programming Guide
---

* This will become a table of contents (this text will be scraped).
{:toc}

<p style="text-align: center;">
  <img src="img/graphx_logo.png"
       title="GraphX Logo"
       alt="GraphX"
       width="65%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

# Overview

GraphX is the new (alpha) Spark API for graphs and graph-parallel computation. At a high-level,
GraphX extends the Spark [RDD](api/core/index.html#org.apache.spark.rdd.RDD) by introducing the
[Resilient Distributed property Graph (RDG)](#property_graph): a directed multigraph with properties
attached to each vertex and edge.  To support graph computation, GraphX exposes a set of fundamental
operators (e.g., [subgraph](#structural_operators), [joinVertices](#join_operators), and
[mapReduceTriplets](#mrTriplets)) as well as an optimized variant of the [Pregel](#pregel) API. In
addition, GraphX includes a growing collection of graph [algorithms](#graph_algorithms) and
[builders](#graph_builders) to simplify graph analytics tasks.

## Background on Graph-Parallel Computation

From social networks to language modeling, the growing scale and importance of
graph data has driven the development of numerous new *graph-parallel* systems
(e.g., [Giraph](http://http://giraph.apache.org) and
[GraphLab](http://graphlab.org)).  By restricting the types of computation that can be
expressed and introducing new techniques to partition and distribute graphs,
these systems can efficiently execute sophisticated graph algorithms orders of
magnitude faster than more general *data-parallel* systems.

<p style="text-align: center;">
  <img src="img/data_parallel_vs_graph_parallel.png"
       title="Data-Parallel vs. Graph-Parallel"
       alt="Data-Parallel vs. Graph-Parallel"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

However, the same restrictions that enable these substantial performance gains
also make it difficult to express many of the important stages in a typical graph-analytics pipeline:
constructing the graph, modifying its structure, or expressing computation that
spans multiple graphs.  As a consequence, existing graph analytics pipelines
compose graph-parallel and data-parallel systems, leading to extensive data
movement and duplication and a complicated programming model.

<p style="text-align: center;">
  <img src="img/graph_analytics_pipeline.png"
       title="Graph Analytics Pipeline"
       alt="Graph Analytics Pipeline"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

The goal of the GraphX project is to unify graph-parallel and data-parallel computation in one
system with a single composable API. The GraphX API enables users to view data both as a graph and
as collections (i.e., RDDs) without data movement or duplication. By incorporating recent advances
in graph-parallel systems, GraphX is able to optimize the execution of graph operations.

## GraphX Replaces the Spark Bagel API

Prior to the release of GraphX, graph computation in Spark was expressed using Bagel, an
implementation of Pregel.  GraphX improves upon Bagel by exposing a richer property graph API, a
more streamlined version of the Pregel abstraction, and system optimizations to improve performance
and reduce memory overhead.  While we plan to eventually deprecate Bagel, we will continue to
support the [Bagel API](api/bagel/index.html#org.apache.spark.bagel.package) and
[Bagel programming guide](bagel-programming-guide.html). However, we encourage Bagel users to
explore the new GraphX API and comment on issues that may complicate the transition from Bagel.

# Getting Started

To get started you first need to import Spark and GraphX into your project, as follows:

{% highlight scala %}
import org.apache.spark._
import org.apache.spark.graphx._
{% endhighlight %}

If you are not using the Spark shell you will also need a Spark context.

# The Property Graph
<a name="property_graph"></a>

The [property graph](api/graphx/index.html#org.apache.spark.graphx.Graph) is a directed multigraph
with user defined objects attached to each vertex and edge.  A directed multigraph is a directed
graph with potentially multiple parallel edges sharing the same source and destination vertex.  The
ability to support parallel edges simplifies modeling scenarios where there can be multiple
relationships (e.g., co-worker and friend) between the same vertices.  Each vertex is keyed by a
*unique* 64-bit long identifier (`VertexId`).  Similarly, edges have corresponding source and
destination vertex identifiers. GraphX does not impose any ordering or constraints on the vertex
identifiers.  The property graph is parameterized over the vertex `VD` and edge `ED` types.  These
are the types of the objects associated with each vertex and edge respectively.

> GraphX optimizes the representation of `VD` and `ED` when they are plain old data-types (e.g.,
> int, double, etc...) reducing the in memory footprint.

In some cases we may wish to have vertices with different property types in the same graph. This can
be accomplished through inheritance.  For example to model users and products as a bipartite graph
we might do the following:

{% highlight scala %}
case class VertexProperty
case class UserProperty extends VertexProperty
  (val name: String)
case class ProductProperty extends VertexProperty
  (val name: String, val price: Double)
// The graph might then have the type:
val graph: Graph[VertexProperty, String]
{% endhighlight %}

Like RDDs, property graphs are immutable, distributed, and fault-tolerant.  Changes to the values or
structure of the graph are accomplished by producing a new graph with the desired changes. The graph
is partitioned across the workers using a range of vertex-partitioning heuristics.  As with RDDs,
each partition of the graph can be recreated on a different machine in the event of a failure.

Logically the property graph corresponds to a pair of typed collections (RDDs) encoding the
properties for each vertex and edge.  As a consequence, the graph class contains members to access
the vertices and edges of the graph:

{% highlight scala %}
val vertices: VertexRDD[VD]
val edges: EdgeRDD[ED]
{% endhighlight %}

The classes `VertexRDD[VD]` and `EdgeRDD[ED]` extend and are optimized versions of `RDD[(VertexId,
VD)]` and `RDD[Edge[ED]]` respectively.  Both `VertexRDD[VD]` and `EdgeRDD[ED]` provide  additional
functionality built around graph computation and leverage internal optimizations.  We discuss the
`VertexRDD` and `EdgeRDD` API in greater detail in the section on [vertex and edge
RDDs](#vertex_and_edge_rdds) but for now they can be thought of as simply RDDs of the form:
`RDD[(VertexId, VD)]` and `RDD[Edge[ED]]`.

### Example Property Graph

Suppose we want to construct a property graph consisting of the various collaborators on the GraphX
project. The vertex property might contain the username and occupation.  We could annotate edges
with a string describing the relationships between collaborators:

<p style="text-align: center;">
  <img src="img/property_graph.png"
       title="The Property Graph"
       alt="The Property Graph"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

The resulting graph would have the type signature:

{% highlight scala %}
val userGraph: Graph[(String, String), String]
{% endhighlight %}

There are numerous ways to construct a property graph from raw files, RDDs, and even synthetic
generators and these are discussed in more detail in the section on
[graph builders](#graph_builders).  Probably the most general method is to use the
[Graph object](api/graphx/index.html#org.apache.spark.graphx.Graph$).  For example the following
code constructs a graph from a collection of RDDs:

{% highlight scala %}
// Assume the SparkContext has already been constructed
val sc: SparkContext
// Create an RDD for the vertices
val users: RDD[(VertexId, (String, String))] =
  sc.parallelize(Array((3, ("rxin", "student")), (7, ("jgonzal", "postdoc")),
                       (5, ("franklin", "prof")), (2, ("istoica", "prof"))))
// Create an RDD for edges
val relationships: RDD[Edge[String]] =
  sc.parallelize(Array(Edge(3, 7, "collab"), Edge(5, 3, "advisor"),
                       Edge(2, 5, "colleague"), Edge(5, 7, "pi"))
// Define a default user in case there are relationship with missing user
val defaultUser = ("John Doe", "Missing")
// Build the initial Graph
val graph = Graph(users, relationships, defaultUser)
{% endhighlight %}

In the above example we make use of the [`Edge`][Edge] case class. Edges have a `srcId` and a
`dstId` corresponding to the source and destination vertex identifiers. In addition, the `Edge`
class contains the `attr` member which contains the edge property.

[Edge]: api/graphx/index.html#org.apache.spark.graphx.Edge

We can deconstruct a graph into the respective vertex and edge views by using the `graph.vertices`
and `graph.edges` members respectively.

{% highlight scala %}
val graph: Graph[(String, String), String] // Constructed from above
// Count all users which are postdocs
graph.vertices.filter { case (id, (name, pos)) => pos == "postdoc"}.count
// Count all the edges where src > dst
graph.edges.filter(e => e.srcId > e.dstId).count
{% endhighlight %}

> Note that `graph.vertices` returns an `VertexRDD[(String, String)]` which extends
> `RDD[(VertexId, (String, String))]` and so we use the scala `case` expression to deconstruct the
> tuple.  On the other hand, `graph.edges` returns an `EdgeRDD` containing `Edge[String]` objects.
> We could have also used the case class type constructor as in the following:
> {% highlight scala %}
graph.edges.filter { case Edge(src, dst, prop) => src < dst }.count
{% endhighlight %}

In addition to the vertex and edge views of the property graph, GraphX also exposes a triplet view.
The triplet view logically joins the vertex and edge properties yielding an
`RDD[EdgeTriplet[VD, ED]]` containing instances of the [`EdgeTriplet`][EdgeTriplet] class. This
*join* can be expressed in the following SQL expression:

[EdgeTriplet]: api/graphx/index.html#org.apache.spark.graphx.EdgeTriplet

{% highlight sql %}
SELECT src.id, dst.id, src.attr, e.attr, dst.attr
FROM edges AS e LEFT JOIN vertices AS src, vertices AS dst
ON e.srcId = src.Id AND e.dstId = dst.Id
{% endhighlight %}

or graphically as:

<p style="text-align: center;">
  <img src="img/triplet.png"
       title="Edge Triplet"
       alt="Edge Triplet"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

The [`EdgeTriplet`][EdgeTriplet] class extends the [`Edge`][Edge] class by adding the `srcAttr` and
`dstAttr` members which contain the source and destination properties respectively. We can use the
triplet view of a graph to render a collection of strings describing relationships between users.

{% highlight scala %}
val graph: Graph[(String, String), String] // Constructed from above
// Use the triplets view to create an RDD of facts.
val facts: RDD[String] =
  graph.triplets.map(et => et.srcAttr._1 + " is the " + et.attr + " of " et.dstAttr)
{% endhighlight %}

# Graph Operators

Just as RDDs have basic operations like `map`, `filter`, and `reduceByKey`, property graphs also
have a collection of basic operators that take user defined functions and produce new graphs with
transformed properties and structure.  The core operators that have optimized implementations are
defined in [`Graph`][Graph] and convenient operators that are expressed as a compositions of the
core operators are defined in [`GraphOps`][GraphOps].  However, thanks to Scala implicits the
operators in `GraphOps` are automatically available as members of `Graph`.  For example, we can
compute the in-degree of each vertex (defined in `GraphOps`) by the following:

[Graph]: api/graphx/index.html#org.apache.spark.graphx.Graph
[GraphOps]: api/graphx/index.html#org.apache.spark.graphx.GraphOps

{% highlight scala %}
val graph: Graph[(String, String), String]
// Use the implicit GraphOps.inDegrees operator
val indDegrees: VertexRDD[Int] = graph.inDegrees
{% endhighlight %}

The reason for differentiating between core graph operations and GraphOps is to be able to support
various graph representations in the future.

## Property Operators

In direct analogy to the RDD `map` operator, the property
graph contains the following:

{% highlight scala %}
def mapVertices[VD2](map: (VertexID, VD) => VD2): Graph[VD2, ED]
def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]
def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]
{% endhighlight %}

Each of these operators yields a new graph with the vertex or edge properties modified by the user
defined `map` function.

> Note that in all cases the graph structure is unaffected. This is a key feature of these operators
> which allows the resulting graph to reuse the structural indices of the original graph. The
> following snippets are logically equivalent, but the first one does not preserve the structural
> indices and would not benefit from the GraphX system optimizations:
> {% highlight scala %}
val newVertices = graph.vertices.map { case (id, attr) => (id, mapUdf(id, attr)) }
val newGraph = Graph(newVertices, graph.edges)
{% endhighlight %}
> Instead, use [`mapVertices`][Graph.mapVertices] to preserve the indices:
> {% highlight scala %}
val newGraph = graph.mapVertices((id, attr) => mapUdf(id, attr))
{% endhighlight %}

[Graph.mapVertices]: api/graphx/index.html#org.apache.spark.graphx.Graph@mapVertices[VD2]((VertexID,VD)⇒VD2)(ClassTag[VD2]):Graph[VD2,ED]

These operators are often used to initialize the graph for a particular computation or project away
unnecessary properties.  For example, given a graph with the out-degrees as the vertex properties
(we describe how to construct such a graph later), we initialize it for PageRank:

{% highlight scala %}
// Given a graph where the vertex property is the out-degree
val inputGraph: Graph[Int, String]
// Construct a graph where each edge contains the weight
// and each vertex is the initial PageRank
val outputGraph: Graph[Double, Double] =
  inputGraph.mapTriplets(et => 1.0 / et.srcAttr).mapVertices(v => 1.0)
{% endhighlight %}

## Structural Operators
<a name="structural_operators"></a>

Currently GraphX supports only a simple set of commonly used structural operators and we expect to
add more in the future.  The following is a list of the basic structural operators.

{% highlight scala %}
def reverse: Graph[VD, ED]
def subgraph(epred: EdgeTriplet[VD,ED] => Boolean,
             vpred: (VertexID, VD) => Boolean): Graph[VD, ED]
def mask[VD2, ED2](other: Graph[VD2, ED2]): Graph[VD, ED]
def groupEdges(merge: (ED, ED) => ED): Graph[VD,ED]
{% endhighlight %}

The [`reverse`][Graph.reverse] operator returns a new graph with all the edge directions reversed.
This can be useful when, for example, trying to compute the inverse PageRank.  Because the reverse
operation does not modify vertex or edge properties or change the number of edges, it can be
implemented efficiently without data-movement or duplication.

[Graph.reverse]: api/graphx/index.html#org.apache.spark.graphx.Graph@reverse:Graph[VD,ED]

The [`subgraph`][Graph.subgraph] operator takes vertex and edge predicates and returns the graph
containing only the vertices that satisfy the vertex predicate (evaluate to true) and edges that
satisfy the edge predicate *and connect vertices that satisfy the vertex predicate*.  The `subgraph`
operator can be used in number of situations to restrict the graph to the vertices and edges of
interest or eliminate broken links. For example in the following code we remove broken links:

[Graph.subgraph]: api/graphx/index.html#org.apache.spark.graphx.Graph@subgraph((EdgeTriplet[VD,ED])⇒Boolean,(VertexID,VD)⇒Boolean):Graph[VD,ED]

{% highlight scala %}
val users: RDD[(VertexId, (String, String))]
val edges: RDD[Edge[String]]
// Define a default user in case there are relationship with missing user
val defaultUser = ("John Doe", "Missing")
// Build the initial Graph
val graph = Graph(users, relationships, defaultUser)
// Remove missing vertices as well as the edges to connected to them
val validGraph = graph.subgraph((id, attr) => attr._2 != "Missing")
{% endhighlight %}

> Note in the above example only the vertex predicate is provided.  The `subgraph` operator defaults
> to `true` if the vertex or edge predicates are not provided.

The [`mask`][Graph.mask] operator also constructs a subgraph by returning a graph that contains the
vertices and edges that are also found in the input graph.  This can be used in conjunction with the
`subgraph` operator to restrict a graph based on the properties in another related graph.  For
example, we might run connected components using the graph with missing vertices and then restrict
the answer to the valid subgraph.

[Graph.mask]: api/graphx/index.html#org.apache.spark.graphx.Graph@mask[VD2,ED2](Graph[VD2,ED2])(ClassTag[VD2],ClassTag[ED2]):Graph[VD,ED]

{% highlight scala %}
// Run Connected Components
val ccGraph = graph.connectedComponents() // No longer contains missing field
// Remove missing vertices as well as the edges to connected to them
val validGraph = graph.subgraph((id, attr) => attr._2 != "Missing")
// Restrict the answer to the valid subgraph
val validCCGraph = ccGraph.mask(validGraph)
{% endhighlight %}

The [`groupEdges`][Graph.groupEdges] operator merges parallel edges (i.e., duplicate edges between
pairs of vertices) in the multigraph.  In many numerical applications, parallel edges can be *added*
(their weights combined) into a single edge thereby reducing the size of the graph.

[Graph.groupEdges]: api/graphx/index.html#org.apache.spark.graphx.Graph@groupEdges((ED,ED)⇒ED):Graph[VD,ED]

## Join Operators
<a name="join_operators"></a>

In many cases it is necessary to join data from external collections (RDDs) with graphs.  For
example, we might have extra user properties that we want to merge with an existing graph or we
might want to pull vertex properties from one graph into another.  These tasks can be accomplished
using the *join* operators. Below we list the key join operators:

{% highlight scala %}
def joinVertices[U](table: RDD[(VertexID, U)])(map: (VertexID, VD, U) => VD)
  : Graph[VD, ED]
def outerJoinVertices[U, VD2](table: RDD[(VertexID, U)])(map: (VertexID, VD, Option[U]) => VD2)
  : Graph[VD2, ED]
{% endhighlight %}

The [`joinVertices`][GraphOps.joinVertices] operator joins the vertices with the input RDD and
returns a new graph with the vertex properties obtained by applying the user defined `map` function
to the result of the joined vertices.  Vertices without a matching value in the RDD retain their
original value.

[GraphOps.joinVertices]: api/graphx/index.html#org.apache.spark.graphx.GraphOps@joinVertices[U](RDD[(VertexID,U)])((VertexID,VD,U)⇒VD)(ClassTag[U]):Graph[VD,ED]

> Note that if the RDD contains more than one value for a given vertex only one will be used.   It
> is therefore recommended that the input RDD be first made unique using the following which will
> also *pre-index* the resulting values to substantially accelerate the subsequent join.
> {% highlight scala %}
val nonUniqueCosts: RDD[(VertexId, Double)]
val uniqueCosts: VertexRDD[Double] =
  graph.vertices.aggregateUsingIndex(nonUnique, (a,b) => a + b)
val joinedGraph = graph.joinVertices(uniqueCosts)(
  (id, oldCost, extraCost) => oldCost + extraCost)
{% endhighlight %}

The more general [`outerJoinVertices`][Graph.outerJoinVertices] behaves similarly to `joinVertices`
except that the user defined `map` function is applied to all vertices and can change the vertex
property type.  Because not all vertices may have a matching value in the input RDD the `map`
function takes an `Option` type.  For example, we can setup a graph for PageRank by initializing
vertex properties with their `outDegree`.

[Graph.outerJoinVertices]: api/graphx/index.html#org.apache.spark.graphx.Graph@outerJoinVertices[U,VD2](RDD[(VertexID,U)])((VertexID,VD,Option[U])⇒VD2)(ClassTag[U],ClassTag[VD2]):Graph[VD2,ED]


{% highlight scala %}
val outDegrees: VertexRDD[Int] = graph.outDegrees
val degreeGraph = graph.outerJoinVertices(outDegrees) { (id, oldAttr, outDegOpt) =>
  outDegOpt match {
    case Some(outDeg) => outDeg
    case None => 0 // No outDegree means zero outDegree
  }
}
{% endhighlight %}

> You may have noticed the multiple parameter lists (e.g., `f(a)(b)`) curried function pattern used
> in the above examples.  While we could have equally written `f(a)(b)` as `f(a,b)` this would mean
> that type inference on `b` would not depend on `a`.  As a consequence, the user would need to
> provide type annotation for the user defined function:
> {% highlight scala %}
val joinedGraph = graph.joinVertices(uniqueCosts,
  (id: VertexId, oldCost: Double, extraCost: Double) => oldCost + extraCost)
{% endhighlight %}


## Neighborhood Aggregation

A key part of graph computation is aggregating information about the neighborhood of each vertex.
For example we might want to know the number of followers each user has or the average age of the
the followers of each user.  Many iterative graph algorithms (e.g., PageRank, Shortest Path, and
connected components) repeatedly aggregate properties of neighboring vertices (e.g., current
PageRank Value, shortest path to the source, and smallest reachable vertex id).

### Map Reduce Triplets (mapReduceTriplets)
<a name="mrTriplets"></a>

[Graph.mapReduceTriplets]: api/graphx/index.html#mapReduceTriplets[A](mapFunc:org.apache.spark.graphx.EdgeTriplet[VD,ED]=&gt;Iterator[(org.apache.spark.graphx.VertexID,A)],reduceFunc:(A,A)=&gt;A,activeSetOpt:Option[(org.apache.spark.graphx.VertexRDD[_],org.apache.spark.graphx.EdgeDirection)])(implicitevidence$10:scala.reflect.ClassTag[A]):org.apache.spark.graphx.VertexRDD[A]

These core (heavily optimized) aggregation primitive in GraphX is the
(`mapReduceTriplets`)[Graph.mapReduceTriplets] operator:

{% highlight scala %}
def mapReduceTriplets[A](
    map: EdgeTriplet[VD, ED] => Iterator[(VertexID, A)],
    reduce: (A, A) => A)
  : VertexRDD[A]
{% endhighlight %}

The (`mapReduceTriplets`)[Graph.mapReduceTriplets] operator takes a user defined map function which
is applied to each triplet and can yield *messages* destined to either (none or both) vertices in
the triplet.  We currently only support messages destined to the source or destination vertex of the
triplet to enable optimized preaggregation.  The user defined `reduce` function combines the
messages destined to each vertex.  The `mapReduceTriplets` operator returns a `VertexRDD[A]`
containing the aggregate message to each vertex.  Vertices that do not receive a message are not
included in the returned `VertexRDD`.

> Note that `mapReduceTriplets takes an additional optional `activeSet` (see API docs) which
> restricts the map phase to edges adjacent to the vertices in the provided `VertexRDD`. Restricting
> computation to triplets adjacent to a subset of the vertices is often necessary in incremental
> iterative computation and is a key part of the GraphX implementation of Pregel.

We can use the `mapReduceTriplets` operator to collect information about adjacent vertices.  For
example if we wanted to compute the average age of followers who are older that each user we could
do the following.

{% highlight scala %}
// Graph with age as the vertex property
val graph: Graph[Double, String] = getFromSomewhereElse()
// Compute the number of older followers and their total age
val olderFollowers: VertexRDD[(Int, Double)] = graph.mapReduceTriplets[(Int, Double)](
  triplet => { // Map Function
    if (triplet.srcAttr > triplet.dstAttr) {
      // Send message to destination vertex containing counter and age
      Iterator((triplet.dstId, (1, triplet.srcAttr)))
    } else {
      // Don't send a message for this triplet
      Iterator.empty
    }
  },
  // Add counter and age
  (a, b) => (a._1 + b._1, a._2 + b._2) // Reduce Function
)
// Divide total age by number of older followers to get average age of older followers
val avgAgeOlderFollowers: VertexRDD[Double] =
  olderFollowers.mapValues { case (count, totalAge) => totalAge / count }
{% endhighlight %}

> Note that the `mapReduceTriplets` operation performs optimally when the messages (and their sums)
> are constant sized (e.g., floats and addition instead of lists and concatenation).  More
> precisely, the result of `mapReduceTriplets` should be sub-linear in the degree of each vertex.

Because it is often necessary to aggregate information about neighboring vertices we also provide an
alternative interface defined in [`GraphOps`][GraphOps]:

{% highlight scala %}
def aggregateNeighbors[A](
    map: (VertexID, EdgeTriplet[VD, ED]) => Option[A],
    reduce: (A, A) => A,
    edgeDir: EdgeDirection)
  : VertexRDD[A]
{% endhighlight %}

The `aggregateNeighbors` operator is implemented directly on top of `mapReduceTriplets` but allows
the user to define the logic in a more vertex centric manner.  Here the `map` function is provided
the vertex to which the message is sent as well as one of the edges and returns the optional message
value.  The `edgeDir` determines whether the `map` function is run on `In`, `Out`, or `All` edges
adjacent to each vertex.

### Computing Degree Information

A common aggregation task is computing the degree of each vertex: the number of edges adjacent to
each vertex.  In the context of directed graphs it often necessary to know the in-degree, out-
degree, and the total degree of each vertex.  The  [`GraphOps`][GraphOps] class contains a
collection of operators to compute the degrees of each vertex.  For example in the following we
compute the max in, out, and total degrees:

{% highlight scala %}
// Define a reduce operation to compute the highest degree vertex
def maxReduce(a: (VertexId, Int), b: (VertexId, Int)): (VertexId, Int) = {
  if (a._2 > b._2) a else b
}
// Compute the max degrees
val maxInDegree: (VertexId, Int)  = graph.inDegrees.reduce(maxReduce)
val maxOutDegree: (VertexId, Int) = graph.outDegrees.reduce(maxReduce)
val maxDegrees: (VertexId, Int)   = graph.degrees.reduce(maxReduce)
{% endhighlight %}


### Collecting Neighbors

In some cases it may be easier to express computation by collecting neighboring vertices and their
attributes at each vertex. This can be easily accomplished using the `collectNeighborIds` and the
`collectNeighbors` operators.

{% highlight scala %}
def collectNeighborIds(edgeDirection: EdgeDirection): VertexRDD[Array[VertexID]] =
def collectNeighbors(edgeDirection: EdgeDirection): VertexRDD[ Array[(VertexID, VD)] ]
{% endhighlight %}

> Note that these operators can be quite costly as they duplicate information and require
> substantial communication.  If possible try expressing the same computation using the
> `mapReduceTriplets` operator directly.

# Pregel API
<a name="pregel"></a>

Graphs are inherently recursive data-structures as properties of a vertices depend on properties of
their neighbors which intern depend on properties of the neighbors of their neighbors.  As a
consequence many important graph algorithms iteratively recompute the properties of each vertex
until a fixed-point condition is reached.  A range of graph-parallel abstractions have been proposed
to express these iterative algorithms.  GraphX exposes a Pregel operator which is a fusion of
the widely used Pregel and GraphLab abstractions.

At a high-level the GraphX variant of the Pregel abstraction is a bulk-synchronous parallel
messaging abstraction constrained to the topology of the graph.  The Pregel operator executes in a
series of super-steps in which vertices receive the sum of their inbound messages from the previous
super-step, compute a new property value, and then send messages to neighboring vertices in the next
super-step.  Vertices that do not receive a message are skipped within a super-step.  The Pregel
operators terminates iteration and returns the final graph when there are no messages remaining.

> Note, unlike more standard Pregel implementations, vertices in GraphX can only send messages to
> neighboring vertices and the message construction is done in parallel using a user defined
> messaging function.  These constraints allow additional optimization within GraphX.

The following is type signature of the Pregel operator as well as a *sketch* of its implementation
(note calls to graph.cache have been removed):

{% highlight scala %}
def pregel[A]
    (initialMsg: A,
     maxIter: Int = Int.MaxValue,
     activeDir: EdgeDirection = EdgeDirection.Out)
    (vprog: (VertexID, VD, A) => VD,
     sendMsg: EdgeTriplet[VD, ED] => Iterator[(VertexID,A)],
     mergeMsg: (A, A) => A)
  : Graph[VD, ED] = {
  // Receive the initial message at each vertex
  var g = mapVertices( (vid, vdata) => vprog(vid, vdata, initialMsg) ).cache()
  // compute the messages
  var messages = g.mapReduceTriplets(sendMsg, mergeMsg)
  var activeMessages = messages.count()
  // Loop until no messages remain or maxIterations is achieved
  var i = 0
  while (activeMessages > 0 && i < maxIterations) {
    // Receive the messages: -----------------------------------------------------------------------
    // Run the vertex program on all vertices that receive messages
    val newVerts = g.vertices.innerJoin(messages)(vprog).cache()
    // Merge the new vertex values back into the graph
    g = g.outerJoinVertices(newVerts) { (vid, old, newOpt) => newOpt.getOrElse(old) }.cache()
    // Send Messages: ------------------------------------------------------------------------------
    // Vertices that didn't receive a message above don't appear in newVerts and therefore don't
    // get to send messages.  More precisely the map phase of mapReduceTriplets is only invoked
    // on edges in the activeDir of vertices in newVerts
    messages = g.mapReduceTriplets(sendMsg, mergeMsg, Some((newVerts, activeDir))).cache()
    activeMessages = messages.count()
    i += 1
  }
  g
}
{% endhighlight %}

Notice that Pregel takes two argument lists (i.e., `graph.pregel(list1)(list2)`).  The first
argument list contains configuration parameters including the initial message, the maximum number of
iterations, and the edge direction in which to send messages (by default along out edges).  The
second argument list contains the user defined functions for receiving messages (the vertex program
`vprog`), computing messages (`sendMsg`), and combining messages `mergeMsg`.

We can use the Pregel operator to express computation such single source shortest path in the
following example.

{% highlight scala %}
val graph: Graph[String, Double] // A graph with edge attributes containing distances
val sourceId: VertexId = 42 // The ultimate source
// Initialize the graph such that all vertices except the root have distance infinity.
val initialGraph = graph.mapVertices((id, _) => if (id == shourceId) 0.0 else Double.PositiveInfinity)
val sssp = initialGraph.pregel(Double.PositiveInfinity)(
  (id, dist, newDist) => math.min(dist, newDist) // Vertex Program
  triplet => {  // Send Message
    if(triplet.srcAttr + triplet.attr < triplet.dstAttr) {
      Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
    } else {
      Iterator.empty
    }
  },
  (a,b) => math.min(a,b) // Merge Message
  )
{% endhighlight %}

# Graph Builders
<a name="graph_builders"></a>

# Vertex and Edge RDDs
<a name="vertex_and_edge_rdds"></a>



# Optimized Representation

This section should give some intuition about how GraphX works and how that affects the user (e.g.,
things to worry about.)

<p style="text-align: center;">
  <img src="img/edge_cut_vs_vertex_cut.png"
       title="Edge Cut vs. Vertex Cut"
       alt="Edge Cut vs. Vertex Cut"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

<p style="text-align: center;">
  <img src="img/vertex_routing_edge_tables.png"
       title="RDD Graph Representation"
       alt="RDD Graph Representation"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>




# Graph Algorithms
<a name="graph_algorithms"></a>

GraphX includes a set of graph algorithms in to simplify analytics. The algorithms are contained in the `org.apache.spark.graphx.lib` package and can be accessed directly as methods on `Graph` via [`GraphOps`][GraphOps]. This section describes the algorithms and how they are used.

## PageRank
<a name="pagerank"></a>

PageRank measures the importance of each vertex in a graph, assuming an edge from *u* to *v* represents an endorsement of *v*'s importance by *u*. For example, if a Twitter user is followed by many others, the user will be ranked highly.

GraphX comes with static and dynamic implementations of PageRank as methods on the [`PageRank` object][PageRank]. Static PageRank runs for a fixed number of iterations, while dynamic PageRank runs until the ranks converge (i.e., stop changing by more than a specified tolerance). [`GraphOps`][GraphOps] allows calling these algorithms directly as methods on `Graph`.

GraphX also includes an example social network dataset that we can run PageRank on. A set of users is given in `graphx/data/users.txt`, and a set of relationships between users is given in `graphx/data/followers.txt`. We compute the PageRank of each user as follows:

[PageRank]: api/graphx/index.html#org.apache.spark.graphx.lib.PageRank$

{% highlight scala %}
// Load the edges as a graph
val graph = GraphLoader.edgeListFile(sc, "graphx/data/followers.txt")
// Run PageRank
val ranks = graph.pageRank(0.0001).vertices
// Join the ranks with the usernames
val users = sc.textFile("graphx/data/users.txt").map { line =>
  val fields = line.split("\\s+")
  (fields(0).toLong, fields(1))
}
val ranksByUsername = users.leftOuterJoin(ranks).map {
  case (id, (username, rankOpt)) => (username, rankOpt.getOrElse(0.0))
}
// Print the result
println(ranksByUsername.collect().mkString("\n"))
{% endhighlight %}

## Connected Components

The connected components algorithm labels each connected component of the graph with the ID of its lowest-numbered vertex. For example, in a social network, connected components can approximate clusters. GraphX contains an implementation of the algorithm in the [`ConnectedComponents` object][ConnectedComponents], and we compute the connected components of the example social network dataset from the [PageRank section](#pagerank) as follows:

[ConnectedComponents]: api/graphx/index.html#org.apache.spark.graphx.lib.ConnectedComponents$

{% highlight scala %}
// Load the graph as in the PageRank example
val graph = GraphLoader.edgeListFile(sc, "graphx/data/followers.txt")
// Find the connected components
val cc = graph.connectedComponents().vertices
// Join the connected components with the usernames
val users = sc.textFile("graphx/data/users.txt").map { line =>
  val fields = line.split("\\s+")
  (fields(0).toLong, fields(1))
}
val ccByUsername = users.join(cc).map { case (id, (username, cc)) =>
  (username, cc)
}
// Print the result
println(ccByUsername.collect().mkString("\n"))
{% endhighlight %}

## Triangle Counting

A vertex is part of a triangle when it has two adjacent vertices with an edge between them. GraphX implements a triangle counting algorithm in the [`TriangleCount` object][TriangleCount] that determines the number of triangles passing through each vertex, providing a measure of clustering. We compute the triangle count of the social network dataset from the [PageRank section](#pagerank). *Note that `TriangleCount` requires the edges to be in canonical orientation (`srcId < dstId`) and the graph to be partitioned using [`Graph#partitionBy`][Graph.partitionBy].*

[TriangleCount]: api/graphx/index.html#org.apache.spark.graphx.lib.TriangleCount$
[Graph.partitionBy]: api/graphx/index.html#org.apache.spark.graphx.Graph@partitionBy(PartitionStrategy):Graph[VD,ED]

{% highlight scala %}
// Load the edges in canonical order and partition the graph for triangle count
val graph = GraphLoader.edgeListFile(sc, "graphx/data/followers.txt", true).partitionBy(RandomVertexCut)
// Find the triangle count for each vertex
val triCounts = graph.triangleCount().vertices
// Join the triangle counts with the usernames
val users = sc.textFile("graphx/data/users.txt").map { line =>
  val fields = line.split("\\s+")
  (fields(0).toLong, fields(1))
}
val triCountByUsername = users.join(triCounts).map { case (id, (username, tc)) =>
  (username, tc)
}
// Print the result
println(triCountByUsername.collect().mkString("\n"))
{% endhighlight %}

## K-Core

## LDA

<p style="text-align: center;">
  <img src="img/tables_and_graphs.png"
       title="Tables and Graphs"
       alt="Tables and Graphs"
       width="50%" />
  <!-- Images are downsized intentionally to improve quality on retina displays -->
</p>

# Examples

Suppose I want to build a graph from some text files, restrict the graph
to important relationships and users, run page-rank on the sub-graph, and
then finally return attributes associated with the top users.  I can do
all of this in just a few lines with GraphX:

{% highlight scala %}
// Connect to the Spark cluster
val sc = new SparkContext("spark://master.amplab.org", "research")

// Load my user data and prase into tuples of user id and attribute list
val users = sc.textFile("hdfs://user_attributes.tsv")
  .map(line => line.split).map( parts => (parts.head, parts.tail) )

// Parse the edge data which is already in userId -> userId format
val followerGraph = Graph.textFile(sc, "hdfs://followers.tsv")

// Attach the user attributes
val graph = followerGraph.outerJoinVertices(users){
  case (uid, deg, Some(attrList)) => attrList
  // Some users may not have attributes so we set them as empty
  case (uid, deg, None) => Array.empty[String]
  }

// Restrict the graph to users which have exactly two attributes
val subgraph = graph.subgraph((vid, attr) => attr.size == 2)

// Compute the PageRank
val pagerankGraph = Analytics.pagerank(subgraph)

// Get the attributes of the top pagerank users
val userInfoWithPageRank = subgraph.outerJoinVertices(pagerankGraph.vertices){
  case (uid, attrList, Some(pr)) => (pr, attrList)
  case (uid, attrList, None) => (pr, attrList)
  }

println(userInfoWithPageRank.top(5))

{% endhighlight %}