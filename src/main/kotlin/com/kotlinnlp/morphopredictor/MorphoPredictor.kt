/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor

import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The predictor of grammatical properties of tokens.
 * It is a neural processor that given an encoded sentence (as list of encoded tokens) calculates a probability
 * distribution of the intents expressed by the sentence, together with the slot classifications associated to the
 * tokens.
 *
 * @property model the frame extractor model
 * @property propagateToInput whether to propagate errors to the input during the backward (default = false)
 * @property id an identifier of this frame extractor (useful when included in a pool, default = 0)
 */
class MorphoPredictor(
  val model: MorphoPredictorModel,
  override val propagateToInput: Boolean = false,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<Map<String, MorphoPredictor.Prediction>>, // OutputType
  List<Map<String, DenseNDArray>>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * The prediction of a grammatical property.
   *
   * @param property the property name
   * @param value the best value of the property (can be null)
   * @param distribution the probability distribution of all the values
   */
  class Prediction(val property: String, val value: GrammaticalProperty?, val distribution: DenseNDArray)

  /**
   * The dropout is not useful for this processor because it has encodings as input and they make sense if used in
   * their original form.
   */
  override val useDropout: Boolean = false

  /**
   * The BiRNN1 encoder.
   */
  private val biRNNEncoder = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN,
    propagateToInput = this.propagateToInput,
    useDropout = false)

  /**
   * The output networks for the grammatical properties prediction, associated by property name.
   */
  private val outputProcessors: Map<String, BatchFeedforwardProcessor<DenseNDArray>> =
    this.model.outputNetworks.mapValues {
      BatchFeedforwardProcessor<DenseNDArray>(model = it.value, propagateToInput = true, useDropout = false)
    }

  /**
   * Calculate the distribution scores of the morphological properties of the tokens.
   *
   * @param input a list of the token encodings of a sentence
   *
   * @return the predictions of the morphological properties, one per token
   */
  override fun forward(input: List<DenseNDArray>): List<Map<String, Prediction>> {

    val contextEncodings: List<DenseNDArray> = this.biRNNEncoder.forward(input)
    val outputMaps: List<MutableMap<String, Prediction>> =
      List(size = input.size, init = { mutableMapOf<String, Prediction>() })

    this.outputProcessors.forEach { (propertyName, processor) ->
      processor.forward(contextEncodings).zip(outputMaps).forEach { (prediction, outputMap) ->

        val bestIndex: Int = prediction.argMaxIndex()

        outputMap[propertyName] = Prediction(
          property = propertyName,
          value = MorphoPredictorModel.propertiesMap.getValue(propertyName).getOrNull(bestIndex),
          distribution = prediction)
      }
    }

    return outputMaps
  }

  /**
   * Execute the backward of the neural components given the output errors.
   *
   * @param outputErrors the output errors as maps of property names to prediction errors
   */
  override fun backward(outputErrors: List<Map<String, DenseNDArray>>) {

    var encodingErrors: List<DenseNDArray>? = null

    this.outputProcessors.forEach { (propertyName, processor) ->

      processor.backward(outputErrors.map { it.getValue(propertyName) })

      if (encodingErrors == null)
        encodingErrors = processor.getInputErrors(copy = true)
      else
        encodingErrors!!.zip(processor.getInputErrors(copy = false)).forEach { (encodingError, processorError) ->
          encodingError.assignSum(processorError)
        }
    }

    encodingErrors!!.forEach { it.assignDiv(this.outputProcessors.size.toDouble()) }

    this.biRNNEncoder.backward(encodingErrors!!)
  }

  /**
   * Get the list of input errors (they are always a copy).
   *
   * @param copy parameter inherited from the [NeuralProcessor] but without effect actually
   *
   * @return the list of input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> = this.biRNNEncoder.getInputErrors(copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of this frame extractor parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.outputProcessors.values.fold(this.biRNNEncoder.getParamsErrors(copy)) { errors, processor ->
      errors + processor.getParamsErrors(copy)
    }
}
