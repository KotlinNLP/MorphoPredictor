/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor

import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.deeplearning.transformers.BERT
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.WordPieceTokenizer

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
  override val id: Int = 0
) : NeuralProcessor<
  Sentence<FormToken>, // InputType
  List<Map<String, MorphoPredictor.Prediction>>, // OutputType
  List<Map<String, DenseNDArray>>, // ErrorsType
  NeuralProcessor.NoInputErrors // InputErrorsType
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
   *
   */
  override val propagateToInput: Boolean = false

  /**
   * The input tokenizer.
   */
  private val tokenizer = WordPieceTokenizer(this.model.bertModel.vocabulary)

  /**
   * The context encoder.
   */
  private val contextEncoder = BERT(model = this.model.bertModel, fineTuning = true, propagateToInput = false)

  /**
   * The output networks for the grammatical properties prediction, associated by property name.
   */
  private val outputProcessors: Map<String, BatchFeedforwardProcessor<DenseNDArray>> =
    this.model.outputNetworks.mapValues {
      BatchFeedforwardProcessor<DenseNDArray>(model = it.value, propagateToInput = true)
    }

  /**
   * The ranges of word-pieces groups of the last forward.
   * This is a support variable for the backward.
   */
  private lateinit var lastWordPiecesRanges: List<IntRange>

  /**
   * Calculate the distribution scores of the morphological properties of the tokens.
   *
   * @param input a list of the token encodings of a sentence
   *
   * @return the predictions of the morphological properties, one per token
   */
  override fun forward(input: Sentence<FormToken>): List<Map<String, Prediction>> {

    val wordPieces: List<String> = this.tokenizer.tokenize(input.tokens.asSequence().map { it.form })
    this.lastWordPiecesRanges = this.tokenizer.groupPieces(wordPieces)

    val piecesVectors: List<DenseNDArray> = this.contextEncoder.forward(wordPieces)
    val contextEncodings: List<DenseNDArray> = this.mergeEncodings(encodings = piecesVectors)

    require(contextEncodings.size == input.tokens.size)

    val outputMaps: List<MutableMap<String, Prediction>> =
      List(size = input.tokens.size, init = { mutableMapOf<String, Prediction>() })

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

    fun splitErrors(groupSize: Int, errors: DenseNDArray): List<DenseNDArray> =
      if (groupSize == 1) {
        listOf(errors)
      } else {
        val divErrors: DenseNDArray = errors.div(groupSize.toDouble())
        List(groupSize) { divErrors }
      }

    // ------------------------

    var contextErrors: List<DenseNDArray>? = null

    this.outputProcessors.forEach { (propertyName, processor) ->

      processor.backward(outputErrors.map { it.getValue(propertyName) })

      if (contextErrors == null)
        contextErrors = processor.getInputErrors(copy = true)
      else
        contextErrors!!.zip(processor.getInputErrors(copy = false)).forEach { (encodingError, processorError) ->
          encodingError.assignSum(processorError)
        }
    }

    contextErrors!!.forEach { it.assignDiv(this.outputProcessors.size.toDouble()) }

    val contextPiecesErrors: List<DenseNDArray> = this.lastWordPiecesRanges.zip(contextErrors!!)
      .flatMap { (range, errors) -> splitErrors(groupSize = range.length, errors = errors) }

    this.contextEncoder.backward(contextPiecesErrors)
  }

  /**
   * Get the list of input errors (they are always a copy).
   *
   * @param copy parameter inherited from the [NeuralProcessor] but without effect actually
   *
   * @return the list of input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of this frame extractor parameters
   */
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.outputProcessors.values.fold(this.contextEncoder.getParamsErrors(copy)) { errors, processor ->
      errors + processor.getParamsErrors(copy)
    }

  /**
   * Get the encodings of the basic words obtained concatenating consecutive word-pieces of a given list.
   * The encoding of each word is the average of the encodings of its word-pieces components.
   *
   * @param encodings the encodings of the given word-pieces
   *
   * @return the encodings merged by words
   */
  private fun mergeEncodings(encodings: List<DenseNDArray>): List<DenseNDArray> =

    this.lastWordPiecesRanges.map { range ->

      if (range.length == 1) {

        encodings[range.first]

      } else {

        val piecesEncodings: List<DenseNDArray> = encodings.slice(range)

        piecesEncodings[0].sum(piecesEncodings[1])
          .also { res -> piecesEncodings.subList(2, piecesEncodings.size).forEach { res.assignSum(it) } }
          .assignDiv(piecesEncodings.size.toDouble())
      }
    }

  /**
   * The length of this range.
   */
  private val IntRange.length get() = this.last - this.first + 1
}
