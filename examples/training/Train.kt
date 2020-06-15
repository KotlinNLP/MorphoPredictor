/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.morphopredictor.MorphoPredictorModel
import com.kotlinnlp.morphopredictor.helpers.Trainer
import com.kotlinnlp.morphopredictor.helpers.Evaluator
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Train a [MorphoPredictorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val analyzer = MorphologicalAnalyzer(dictionary = parsedArgs.morphoDictionaryPath.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(File(it)))
  })
  val trainingDataset: Dataset = parsedArgs.trainingSetPath.let {
    println("Loading training dataset from '$it'...")
    Dataset.fromFile(filePath = it, analyzer = analyzer)
  }
  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(filePath = it, analyzer = analyzer)
  }

  val encoderModel: TokensEncoderModel<FormToken, MorphoSentence<FormToken>> =
    buildTokensEncoderModel(parsedArgs = parsedArgs, trainingDataset = trainingDataset)
  val model = MorphoPredictorModel(tokenEncodingSize = encoderModel.tokenEncodingSize, hiddenSize = 200)

  println()
  println("Training examples: ${trainingDataset.examples.size}.")
  println("Validation examples: ${validationDataset.examples.size}.")

  Trainer(
    model = model,
    encoderModel = encoderModel,
    dataset = trainingDataset,
    analyzer = analyzer,
    modelFilename = parsedArgs.modelPath,
    epochs = parsedArgs.epochs,
    encoderUpdateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    evaluator = Evaluator(model = model, encoderModel = encoderModel, dataset = validationDataset),
    saveWholeModel = true
  ).train()
}
