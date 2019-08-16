/* Copyright 2019-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.morphopredictor.helpers.dataset

import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalProperty
import com.kotlinnlp.linguisticdescription.morphology.properties.GrammaticalPropertyFactory
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import java.io.File
import java.lang.StringBuilder
import com.kotlinnlp.linguisticdescription.sentence.Sentence as LDSentence

/**
 * A dataset that can be read from a JSONL file.
 *
 * @property examples the list of examples
 */
data class Dataset(val examples: List<Example>) {

  /**
   * Raised when an example is not valid.
   *
   * @param index the example index
   */
  class InvalidExample(index: Int) : RuntimeException("Example #$index")

  /**
   * An example of the dataset.
   *
   * @property sentence the sentence of the example
   */
  data class Example(val sentence: Sentence) {

    /**
     * The sentence of the example.
     *
     * @property tokens the tokens that compose the sentence
     * @property position the position in the text
     * @property morphoAnalysis the morphological analysis of the sentence
     */
    data class Sentence(
      override val tokens: List<Token>,
      override val position: Position,
      override val morphoAnalysis: MorphologicalAnalysis
    ) : RealSentence<RealToken>, MorphoSentence<RealToken>

    /**
     * A token of the sentence.
     *
     * @property form the token form
     * @property position the position in the sentence
     * @property lemma the lemma
     * @property properties the grammatical properties associated by name
     */
    data class Token(
      override val form: String,
      override val position: Position,
      val lemma: String,
      val properties: Map<String, GrammaticalProperty>
    ) : RealToken {

      companion object {

        /**
         * Build tokens from a JSON object.
         *
         * @param obj the JSON object representing a token with its morphological components
         * @param index the index of the token within the tokens of the sentence
         * @param startChar the index of the first char in the text
         *
         * @return the tokens built from the components of the given JSON object
         */
        internal fun fromJson(obj: JsonObject, index: Int, startChar: Int): List<Token> {

          var start: Int = startChar
          val morphologies: List<JsonObject> = obj.array("morphologies")!!
          val bestMorpho: JsonObject = morphologies.singleOrNull() ?: morphologies.first { it.containsKey("best") }
          val components: List<JsonObject> = bestMorpho.array("components")!!

          return components.map { component ->

            val form: String = if (components.size == 1) obj.string("form")!! else component.string("lemma")!!
            val end: Int = start + form.lastIndex

            Token(
              form = form,
              position = Position(index = index, start = start, end = end),
              lemma = component.string("lemma")!!,
              properties = component.obj("properties")!!.mapValues {
                GrammaticalPropertyFactory(propertyName = it.key, valueAnnotation = it.value as String)
              }
            ).also {
              start = end + 2 // one space follows
            }
          }
        }
      }
    }
  }

  companion object {

    /**
     * A temporary sentence used to analyze the morphology of an input sentence.
     *
     * @property tokens the tokens that compose the sentence
     */
    private class TmpSentence(override val tokens: List<RealToken>, override val position: Position)
      : RealSentence<RealToken>

    /**
     * Load a [Dataset] from a JSONL file, containing lines with the following template:
     *
     *  {
     *    "text: "",
     *    "tokens": [
     *      {
     *        "form": "",
     *        "morphologies": {
     *          "best": true // optional
     *          "components": [
     *            {
     *              "lemma": "",
     *              "pos": "",
     *              "properties": { // each property is optional
     *                "mood": "",
     *                "tense": "",
     *                "gender": "",
     *                "number": "",
     *                "person": "",
     *                "case": "",
     *                "degree": ""
     *              }
     *            }
     *          ]
     *        }
     *      }
     *    ]
     *  }
     *
     * @param filePath the file path of the JSONL file
     *
     * @return the dataset read from the given file
     */
    fun fromFile(filePath: String, analyzer: MorphologicalAnalyzer): Dataset {

      val examples: MutableList<Example> = mutableListOf()
      var lineIndex = 0

      File(filePath).forEachLine { line ->

        val jsonExample: JsonObject = Parser().parse(StringBuilder(line)) as JsonObject

        try {

          var startChar = 0

          val tokens: List<Example.Token> = jsonExample.array<JsonObject>("tokens")!!.withIndex().flatMap { (i, obj) ->
            Example.Token.fromJson(obj = obj, index = i, startChar = startChar).also {
              startChar = it.last().position.end + 2 // one space follows
            }
          }

          val position = Position(index = 0, start = 0, end = tokens.last().position.end)
          val sentence = Example.Sentence(
            tokens = tokens,
            position = position,
            morphoAnalysis = analyzer.analyze(TmpSentence(tokens = tokens, position = position)))

          examples.add(Example(sentence))

        } catch (e: RuntimeException) {
          throw InvalidExample(lineIndex)
        }

        lineIndex++
      }

      return Dataset(examples)
    }
  }
}
