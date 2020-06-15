/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor

import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * Predict the morphological properties of tokens.
 *
 * @param model the text morphological predictor model
 */
class TextMorphoPredictor(val model: TextMorphoPredictorModel) {

  /**
   * A morphological predictor built with the given [model].
   */
  internal val predictor = MorphoPredictor(this.model.morphoPredictor)

  /**
   * The encoder of the input tokens.
   */
  private val encoder: TokensEncoder<FormToken, MorphoSentence<FormToken>> = this.model.tokensEncoder.buildEncoder()

  /**
   * Predict the morphological properties of the tokens of a given sentence.
   *
   * @param sentence a sentence of form tokens
   *
   * @return the morphologies predicted, one per token, as map of property names to predictions
   */
  fun predict(sentence: MorphoSentence<FormToken>): List<Map<String, MorphoPredictor.Prediction>> {

    val tokensEncodings: List<DenseNDArray> = this.encoder.forward(sentence)

    return this.predictor.forward(tokensEncodings)
  }
}
