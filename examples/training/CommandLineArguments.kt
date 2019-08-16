/* Copyright 2019-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
internal class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The number of training epochs (default = 10).
   */
  val epochs: Int by parser.storing(
    "--epochs",
    help="the number of training epochs (default = 10)"
  ) { toInt() }.default(10)

  /**
   * The file path in which to serialize the model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the file path in which to serialize the model"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  )

  /**
   * The file path of the lexicon dictionary.
   */
  val lexiconDictionaryPath: String? by parser.storing(
    "-x",
    "--lexicon",
    help="the file path of the lexicon dictionary"
  ).default { null }

  /**
   * The file path of the serialized direct char language model.
   */
  val directLanguageModelPath: String by parser.storing(
    "--direct-lm",
    help="the file path of the serialized direct char language model"
  )

  /**
   * The file path of the serialized reversed char language model.
   */
  val reversedLanguageModelPath: String by parser.storing(
    "--reversed-lm",
    help="the file path of the serialized reversed char language model"
  )

  /**
   * The file path of the training dataset.
   */
  val trainingSetPath: String by parser.storing(
    "-t",
    "--training-set-path",
    help="the file path of the training dataset"
  )

  /**
   * The file path of the validation dataset.
   */
  val validationSetPath: String by parser.storing(
    "-v",
    "--validation-set-path",
    help="the file path of the validation dataset"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
