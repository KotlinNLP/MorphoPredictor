/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor.helpers

import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.morphopredictor.MorphoPredictor
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphopredictor.MorphoPredictorModel
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.helpers.Trainer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.Shuffler
import java.io.File
import java.io.FileOutputStream

/**
 * A helper to train a [MorphoPredictorModel].
 *
 * @param model the model to train
 * @param analyzer a morphological analyzer
 * @param modelFilename the path of the file in which to save the serialized trained model
 * @param epochs the number of training epochs
 * @param updateMethod the update method to optimize the model parameters
 * @param evaluator the helper for the evaluation (default null)
 * @param shuffler used to shuffle the examples before each epoch (with pseudo random by default)
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class Trainer(
  private val model: MorphoPredictorModel,
  modelFilename: String,
  dataset: Dataset,
  private val analyzer: MorphologicalAnalyzer,
  epochs: Int,
  updateMethod: UpdateMethod<*> = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  evaluator: Evaluator,
  shuffler: Shuffler = Shuffler(),
  verbose: Boolean = true
) : Trainer<Dataset.Example>(
  modelFilename = modelFilename,
  optimizers = listOf(ParamsOptimizer(updateMethod)),
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
  private val predictor = MorphoPredictor(this.model)

  /**
   * Learn from an example (forward + backward).
   *
   * @param example an example to train the model with
   */
  override fun learnFromExample(example: Dataset.Example) {

    @Suppress("UNCHECKED_CAST")
    val output: List<Map<String, MorphoPredictor.Prediction>> =
      this.predictor.forward(example.sentence as Sentence<FormToken>)

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
  }

  /**
   * Accumulate the errors of the model resulting after the call of [learnFromExample].
   */
  override fun accumulateErrors() {

    // Note: optimized copies
    this.optimizers.single().accumulate(this.predictor.getParamsErrors(copy = false), copy = false)
  }

  /**
   * Dump the model to file.
   */
  override fun dumpModel() {
    this.model.dump(FileOutputStream(File(this.modelFilename)))
  }
}
