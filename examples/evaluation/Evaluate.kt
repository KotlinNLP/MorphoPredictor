/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.morphopredictor.MorphoPredictorModel
import com.kotlinnlp.morphopredictor.helpers.Evaluator
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.simplednn.helpers.Statistics
import com.kotlinnlp.utils.Timer
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate a [MorphoPredictorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val analyzer = MorphologicalAnalyzer(dictionary = parsedArgs.morphoDictionaryPath.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(File(it)))
  })

  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(filePath = it, analyzer = analyzer)
  }

  val model: MorphoPredictorModel = parsedArgs.modelPath.let {
    println("Loading text morphological predictor model from '$it'...")
    MorphoPredictorModel.load(FileInputStream(File(it)))
  }

  val evaluator = Evaluator(model = model, dataset = validationDataset)

  println("\nStart validation on %d examples".format(validationDataset.examples.size))

  val timer = Timer()
  val stats: Statistics = evaluator.evaluate()

  println("Elapsed time: %s".format(timer.formatElapsedTime()))
  println()
  println("Statistics\n$stats")
}
