/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor.helpers

import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.morphopredictor.MorphoPredictor
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphopredictor.MorphoPredictorModel
import com.kotlinnlp.morphopredictor.TextMorphoPredictorModel
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.Shuffler
import java.io.File
import java.io.FileOutputStream

/**
 * A helper to train a [MorphoPredictorModel].
 *
 * @param model the model to train
 * @param encoderModel the model of a tokens encoder to encode the input
 * @param analyzer a morphological analyzer
 * @param modelFilename the path of the file in which to save the serialized trained model
 * @param epochs the number of training epochs
 * @param predictorUpdateMethod the update method to optimize the morphological predictor model parameters
 * @param encoderUpdateMethod the update method for the parameters of the tokens encoder (null if must not be trained)
 * @param evaluator the helper for the evaluation (default null)
 * @param saveWholeModel whether to save the whole model as [TextMorphoPredictorModel]
 * @param shuffler used to shuffle the examples before each epoch (with pseudo random by default)
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class Trainer(
  private val model: MorphoPredictorModel,
  modelFilename: String,
  dataset: Dataset,
  private val encoderModel: TokensEncoderModel<FormToken, MorphoSentence<FormToken>>,
  private val analyzer: MorphologicalAnalyzer,
  epochs: Int,
  predictorUpdateMethod: UpdateMethod<*> = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  encoderUpdateMethod: UpdateMethod<*>? = null,
  evaluator: Evaluator,
  private val saveWholeModel: Boolean,
  shuffler: Shuffler = Shuffler(),
  verbose: Boolean = true
) : Trainer<Dataset.Example>(
  modelFilename = modelFilename,
  optimizers = if (encoderUpdateMethod != null)
    listOf(ParamsOptimizer(predictorUpdateMethod), ParamsOptimizer(encoderUpdateMethod))
  else
    listOf(ParamsOptimizer(predictorUpdateMethod)),
  examples = dataset.examples,
  epochs = epochs,
  batchSize = 1,
  evaluator = evaluator,
  shuffler = shuffler,
  verbose = verbose
) {

  /**
   * A morphological predictor built with the given [model].
   */
  private val predictor = MorphoPredictor(model = this.model, propagateToInput = encoderUpdateMethod != null)

  /**
   * The encoder of the input tokens.
   */
  private val encoder: TokensEncoder<FormToken, MorphoSentence<FormToken>> = encoderModel.buildEncoder()

  /**
   * The optimizer of the [model] parameters.
   */
  private val predictorOptimizer: ParamsOptimizer = this.optimizers[0]

  /**
   * The optimizer of the tokens encoder.
   * It is null if it must not be trained.
   */
  private val encoderOptimizer: ParamsOptimizer? = this.optimizers.getOrNull(1)

  /**
   * Learn from an example (forward + backward).
   *
   * @param example an example to train the model with
   */
  override fun learnFromExample(example: Dataset.Example) {

    @Suppress("UNCHECKED_CAST")
    val tokensEncodings: List<DenseNDArray> = this.encoder.forward(example.sentence as MorphoSentence<FormToken>)
    val output: List<Map<String, MorphoPredictor.Prediction>> = this.predictor.forward(tokensEncodings)

    val outputErrors: List<Map<String, DenseNDArray>> =
      example.sentence.tokens.zip(output).map { (token, predictions) ->
        predictions.mapValues { (propertyName, prediction) ->

          val propertyValues: List<GrammaticalProperty> = MorphoPredictorModel.propertiesMap.getValue(propertyName)
          val goldValue: GrammaticalProperty? = token.properties[propertyName]

          // Handle the 'null' value
          val goldIndex: Int = goldValue?.let { propertyValues.indexOf(it) } ?: propertyValues.size

          SoftmaxCrossEntropyCalculator.calculateErrors(output = prediction.distribution, goldIndex = goldIndex)
        }
      }

    this.predictor.backward(outputErrors)
    this.encoderOptimizer?.also { this.encoder.backward(this.predictor.getInputErrors(copy = false)) }
  }

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  override fun accumulateErrors() {

    // Note: optimized copies
    this.predictorOptimizer.accumulate(this.predictor.getParamsErrors(copy = false), copy = false)
    this.encoderOptimizer?.accumulate(this.encoder.getParamsErrors(copy = false), copy = false)
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {

    if (this.saveWholeModel)
      TextMorphoPredictorModel(tokensEncoder = encoderModel, morphoPredictor = this.model)
        .dump(FileOutputStream(File(this.modelFilename)))
    else
      this.model.dump(FileOutputStream(File(this.modelFilename)))
  }
}
