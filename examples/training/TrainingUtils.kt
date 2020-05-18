/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphopredictor.helpers.dataset.Dataset
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatFeedforwardMerge
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.charlm.CharLMEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.morpho.FeaturesCollector
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.MirrorConverter
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import java.io.FileInputStream

/**
 * Build an [EnsembleTokensEncoderModel] composed by a morphological encoder and a char based language model.
 *
 * @param parsedArgs the parsed command line arguments
 * @param trainingDataset the training dataset
 *
 * @return a new tokens encoder model
 */
internal fun buildTokensEncoderModel(
  parsedArgs: CommandLineArguments,
  trainingDataset: Dataset
): TokensEncoderModel<FormToken, MorphoSentence<FormToken>> {

  val lexiconDictionary = parsedArgs.lexiconDictionaryPath?.let {
    println("Loading lexicon from '$it'...")
    LexiconDictionary.load(it)
  }
  val featuresDictionary = FeaturesCollector(
    lexicalDictionary = lexiconDictionary,
    sentences = trainingDataset.examples.map { it.sentence }
  ).collect()
  val directCharLM = parsedArgs.directLanguageModelPath.let {
    println("Loading direct char LM from '$it'...")
    CharLM.load(FileInputStream(it))
  }
  val reversedCharLM = parsedArgs.reversedLanguageModelPath.let {
    println("Loading direct char LM from '$it'...")
    CharLM.load(FileInputStream(it))
  }

  return EnsembleTokensEncoderModel(
    components = listOf(
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(
          model = CharLMEncoderModel(dirCharLM = directCharLM, revCharLM = reversedCharLM),
          converter = CastConverter()),
        trainable = false),
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(
          model = MorphoEncoderModel(
            lexiconDictionary = lexiconDictionary,
            featuresDictionary = featuresDictionary,
            tokenEncodingSize = 100,
            activation = null),
          converter = MirrorConverter()),
        trainable = true)
    ),
    outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = 100)
  )
}

/**
 * The sentence converter that casts a morpho sentence of form tokens to a generic sentence of form tokens.
 */
private class CastConverter : SentenceConverter<FormToken, MorphoSentence<FormToken>, FormToken, Sentence<FormToken>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Return the same sentence given in input casted to a generic sentence.
   *
   * @param sentence the input sentence
   *
   * @return the same sentence given in input casted to a generic sentence
   */
  override fun convert(sentence: MorphoSentence<FormToken>): Sentence<FormToken> = sentence
}
