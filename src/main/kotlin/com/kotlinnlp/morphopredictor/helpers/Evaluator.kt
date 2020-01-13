/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor.helpers

import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphopredictor.MorphoPredictor
import com.kotlinnlp.morphopredictor.MorphoPredictorModel
import com.kotlinnlp.morphopredictor.TextMorphoPredictor
import com.kotlinnlp.morphopredictor.TextMorphoPredictorModel
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.simplednn.helpers.Evaluator
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * A helper to evaluate a [MorphoPredictorModel].
 *
 * @param model the model to evaluate
 * @param encoderModel the tokens encoder model
 * @param dataset the validation dataset
 * @param verbose whether to print info about the validation progress (default = true)
 */
class Evaluator(
  model: MorphoPredictorModel,
  encoderModel: TokensEncoderModel<FormToken, MorphoSentence<FormToken>>,
  dataset: Dataset,
  verbose: Boolean = true
) : Evaluator<Dataset.Example, Statistics>(
  examples = dataset.examples,
  verbose = verbose
) {

  /**
   * A morphological predictor built with the given model.
   */
  private val predictor = TextMorphoPredictor(
    model = TextMorphoPredictorModel(morphoPredictor = model, tokensEncoder = encoderModel))

  /**
   * The validation statistics.
   */
  override val stats = Statistics(properties = MorphoPredictorModel.propertiesMap.mapValues { MetricCounter() })

  /**
   * Evaluate the model with a single example.
   *
   * @param example the example to validate the model with
   */
  override fun evaluate(example: Dataset.Example) {

    @Suppress("UNCHECKED_CAST")
    val output: List<Map<String, MorphoPredictor.Prediction>> =
      this.predictor.predict(example.sentence as MorphoSentence<FormToken>)

    example.sentence.tokens.zip(output).forEach { (token, tokenPredictions) ->
      tokenPredictions.forEach { (propertyName, prediction) ->

        val metric: MetricCounter = this.stats.properties.getValue(propertyName)
        val goldValue: GrammaticalProperty? = token.properties[propertyName]

        if (prediction.value == goldValue)
          prediction.value?.also { metric.truePos++ }
        else
          prediction.value?.also { metric.falseNeg++ } ?: metric.falsePos++
      }
    }

    this.stats.accuracy = this.stats.properties.values.asSequence().map { it.f1Score }.average()
  }
}
