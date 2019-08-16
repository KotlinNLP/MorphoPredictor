/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor.helpers

import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * Evaluation statistics.
 *
 * @property properties a map of grammatical properties names to the related metrics
 */
data class Statistics(val properties: Map<String, MetricCounter>) : Statistics() {

  override fun reset() {
    this.properties.values.forEach { it.reset() }
  }

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String {

    val maxIntentLen: Int = this.properties.keys.maxBy { it.length }!!.length + 2 // include apexes

    return """
    - Overall accuracy: ${"%.2f%%".format(100.0 * this.accuracy)}
    - Properties accuracy:
      ${this.properties.entries.joinToString("\n      ") { "%-${maxIntentLen}s : %s".format("`${it.key}`", it.value) }}
    """
      .removePrefix("\n")
      .trimIndent()
  }
}
