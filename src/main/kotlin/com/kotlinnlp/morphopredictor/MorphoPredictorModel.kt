/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor

import com.kotlinnlp.linguisticdescription.morphology.properties.*
import com.kotlinnlp.linguisticdescription.morphology.properties.Number as NumberProp
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatMerge
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [MorphoPredictor] parameters.
 *
 * @param tokenEncodingSize the size of the tokens encodings
 * @param hiddenSize the size of the hidden layer of the BiRNNs
 * @param hiddenActivation the activation of the hidden layer of the BiRNNs
 * @param recurrentConnectionType the connection type of the recurrent layer of the BiRNNs
 */
class MorphoPredictorModel(
  internal val tokenEncodingSize: Int,
  hiddenSize: Int,
  hiddenActivation: ActivationFunction? = Tanh(),
  recurrentConnectionType: LayerType.Connection = LayerType.Connection.LSTM
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Associate each grammatical property to its possible values.
     */
    internal val propertiesMap: Map<String, List<GrammaticalProperty>> = mapOf(
      "mood" to Mood.values().toList(),
      "tense" to Tense.values().toList(),
      "gender" to Gender.values().toList(),
      "number" to NumberProp.values().toList(),
      "person" to Person.values().toList(),
      "case" to GrammaticalCase.values().toList(),
      "degree" to Degree.values().toList()
    )

    /**
     * Read a [MorphoPredictorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [MorphoPredictorModel]
     *
     * @return the [MorphoPredictorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): MorphoPredictorModel = Serializer.deserialize(inputStream)
  }

  /**
   * The BiRNN model for the input encoding.
   */
  val biRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = this.tokenEncodingSize,
    hiddenSize = hiddenSize,
    hiddenActivation = hiddenActivation,
    recurrentConnectionType = recurrentConnectionType,
    dropout = 0.0, // the input is an encoding, it makes sense as complete numerical vector
    outputMergeConfiguration = ConcatMerge())

  /**
   * The output networks for the grammatical properties prediction, associated by property name.
   */
  val outputNetworks: Map<String, StackedLayersParameters> = propertiesMap.mapValues {
    StackedLayersParameters(
      LayerInterface(
        size = this.biRNN.outputSize,
        type = LayerType.Input.Dense),
      LayerInterface(
        size = it.value.size + 1, // the 'null' output is included as last index
        connectionType = LayerType.Connection.Feedforward,
        activationFunction = Softmax())
    )
  }

  /**
   * Serialize this [MorphoPredictorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [MorphoPredictorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
