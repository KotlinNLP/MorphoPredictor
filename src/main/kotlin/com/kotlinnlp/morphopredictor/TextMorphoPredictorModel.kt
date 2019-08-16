/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor

import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [TextMorphoPredictor] model.
 *
 * @property morphoPredictor the model of the morphological predictor
 * @property tokensEncoder the model of a tokens encoder to encode the input
 */
class TextMorphoPredictorModel(
  val morphoPredictor: MorphoPredictorModel,
  val tokensEncoder: TokensEncoderModel<FormToken, MorphoSentence<FormToken>>
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [TextMorphoPredictorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [TextMorphoPredictorModel]
     *
     * @return the [TextMorphoPredictorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): TextMorphoPredictorModel = Serializer.deserialize(inputStream)
  }

  /**
   * Check requirements.
   */
  init {
    require(this.morphoPredictor.tokenEncodingSize == this.tokensEncoder.tokenEncodingSize) {
      "The tokens encoding size of the TokensEncoder must be compatible with the MorphoPredictor."
    }
  }

  /**
   * Serialize this [TextMorphoPredictorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [TextMorphoPredictorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
