/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package predict

import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.morphopredictor.*
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Extract frames from a text.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val tokenizer: NeuralTokenizer = parsedArgs.tokenizerModelPath.let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizer(NeuralTokenizerModel.load(FileInputStream(File(it))))
  }

  val analyzer = MorphologicalAnalyzer(dictionary = parsedArgs.morphoDictionaryPath.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(File(it)))
  })

  val model: MorphoPredictorModel = parsedArgs.modelPath.let {
    println("Loading morphological predictor model from '$it'...")
    MorphoPredictorModel.load(FileInputStream(File(it)))
  }

  val predictor = MorphoPredictor(model)

  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty()) {

      break

    } else {

      tokenizer.tokenize(inputText).forEach { sentence ->

        @Suppress("UNCHECKED_CAST")
        val inputSentence = InputSentence(
          tokens = sentence.tokens,
          morphoAnalysis = analyzer.analyze(sentence as RealSentence<RealToken>))

        val output: List<Map<String, MorphoPredictor.Prediction>> = predictor.forward(inputSentence)

        println()
        printResults(sentence = inputSentence, output = output)
      }
    }
  }

  println("\nExiting...")
}

/**
 * The input sentence for a prediction.
 *
 * @property tokens the tokens that compose the sentence
 * @property morphoAnalysis the morphological analysis of the sentence
 */
private class InputSentence(
  override val tokens: List<FormToken>,
  override val morphoAnalysis: MorphologicalAnalysis
) : MorphoSentence<FormToken>

/**
 * Read a value from the standard input.
 *
 * @return the string read
 */
private fun readValue(): String {

  print("\nPredict the tokens morphologies from a text (empty to exit): ")

  return readLine()!!
}

/**
 * Print the prediction results to the standard output.
 *
 * @param sentence the input sentence
 * @param output the predictions, one per token
 */
private fun printResults(sentence: Sentence<FormToken>, output: List<Map<String, MorphoPredictor.Prediction>>) {

  sentence.tokens.zip(output).forEach { (token, predictions) ->
    println("${token.form}: ${predictions.values.mapNotNull { it.value }.joinToString(" ") { it.annotation }}")
  }
}
