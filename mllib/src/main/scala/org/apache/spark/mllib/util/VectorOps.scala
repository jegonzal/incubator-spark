/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.util

import org.jblas.DoubleMatrix


class DenseVector(data: Array[Double]) {

  def norm: Double = new DoubleMatrix(data).norm2

  def +(other: Array[Double]): Array[Double] = {
    assert(data.size == other.size)
    val thisMat = new DoubleMatrix(data)
    val otherMat = new DoubleMatrix(other)
    thisMat.add(otherMat).data
  }

  def -(other: Array[Double]): Array[Double] = {
    assert(data.size == other.size)
    val thisMat = new DoubleMatrix(data)
    val otherMat = new DoubleMatrix(other)
    thisMat.sub(otherMat).data
  }

  def /(other: Array[Double]): Array[Double] = {
    assert(data.size == other.size)
    val thisMat = new DoubleMatrix(data)
    val otherMat = new DoubleMatrix(other)
    thisMat.div(otherMat).data
  }

  def /(scalar: Double): Array[Double] = {
    val thisMat = new DoubleMatrix(data)
    thisMat.div(scalar).data
  }

  def *(scalar: Double): Array[Double] = {
    val thisMat = new DoubleMatrix(data)
    thisMat.mul(scalar).data
  }


}


object VectorOps {
  def l2dist(a: Array[Double], b: Array[Double]): Double = {
    assert(a.size == b.size)
    var i = 0
    var sum = 0.0
    while (i < a.size) {
      val dist = a(i) - b(i)
      sum += dist * dist
      i += 1
    }
    math.sqrt(sum)
  }

  def norm(a: Array[Double]): Double = { new DoubleMatrix(a).norm2 }

  implicit def toDenseVector(ar: Array[Double]): DenseVector = { new DenseVector(ar) }

} // end of VectorOPs

