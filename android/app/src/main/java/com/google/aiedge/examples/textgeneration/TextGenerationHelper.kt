/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.aiedge.examples.textgeneration

import android.app.ActivityManager
import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.nio.FloatBuffer
import android.os.Debug
import android.os.SystemClock.sleep
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import androidx.core.content.ContextCompat.getSystemService
import kotlinx.coroutines.flow.update
import java.nio.IntBuffer
import kotlin.concurrent.thread
import kotlin.math.max

class TextGenerationHelper(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var tokenizer: GPT2Tokenizer? = null
    init {
        val encoder  = loadEncoder(context)
        val decoder  = encoder.entries.associateBy({ it.value }, { it.key })
        val bpeRanks = loadBpeRanks(context)
        tokenizer = GPT2Tokenizer(encoder, decoder, bpeRanks)

        initClassifier(Model.LocalModel)
    }

    // ----------------- Model Metrics ---------------------------
    private val _modelInfo = MutableStateFlow(ModelInfo())
    val modelInfo: StateFlow<ModelInfo> get() = _modelInfo
    private var _tokensPerSecond: Float = 0f

    // ------------------ MEMORY ------------------------------
    private val _memoryUsage = MutableStateFlow<Pair<Long, Long>>(Pair(0L,0L))
    val memoryUsage: StateFlow<Pair<Long, Long>> get() = _memoryUsage
    private var _totalMemGb: Long = 0L
    private var _initMem: Long = 0L

    private val _textState = MutableStateFlow("")
    val textState: StateFlow<String> get() = _textState

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    var completableDeferred: CompletableDeferred<Unit>? = null

    fun initClassifier(model: Model = Model.LocalModel) {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(info)
        _totalMemGb = (info.totalMem / (1024 * 1024 * 1024))+1

        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, model.fileName)
            _initMem = tfliteBuffer.capacity().toLong()
            Log.i(TAG, "LiteRT buffer criado a partir de ${model.fileName}")
            Interpreter(tfliteBuffer, Interpreter.Options())
        } catch (e: Exception) {
            Log.e(TAG, "Falha ao criar LiteRT a partir de ${model.fileName}: ${e.message}", e)
            null
        }
    }

    fun stopClassify() {
        interpreter?.close()
        interpreter = null
    }

    fun getTokenizeInput(inputText: String): Map<String, Array<IntArray>> {
        val localTokenizer = tokenizer ?: run {
            return mapOf(
                "inputs" to arrayOf(listOf(0).toIntArray())
            )
        }
        val tokens =  localTokenizer.encode(inputText).toIntArray() 
        val attentionMaskArray = arrayOf(IntArray(tokens.size) { 1 })

        return mapOf(
            "input_ids" to arrayOf(tokens),
            "attention_mask" to attentionMaskArray
        )
    }

    suspend fun infer(inputText: String) {
        return withContext(Dispatchers.IO) {
            var peakUsageBytes = 0L
            var monitoring = true
            var runtime = Runtime.getRuntime()

            val monitorThread = thread(start = true) {
                while (monitoring) {
                    val jvmHeap =  runtime.totalMemory() - runtime.freeMemory()
                    val totalUsage = jvmHeap + _initMem
                    peakUsageBytes = max(peakUsageBytes, totalUsage)
                    try {
                        Thread.sleep(50)
                    } catch (e: InterruptedException) {
                        break
                    }
                }
            }
            val inferThread = thread(start = true) {
                _infer(inputText)
                monitoring = false
            }

            monitorThread.join()
            inferThread.join()

            val peakUsageMb = peakUsageBytes / (1024 * 1024)
            _memoryUsage.tryEmit(Pair(peakUsageMb, _totalMemGb))
        }
    }

    fun updateModelMetric(key: String, value: Any) {
        _modelInfo.update { current ->
            val updated = when (key) {
                "inferenceTime" -> current.copy(inferenceTime = value as? Long ?: current.inferenceTime)
                "trainingTime" -> current.copy(trainingTime = value as? Long ?: current.trainingTime)
                "loss" -> current.copy(loss = value as? Float ?: current.loss)
                "tokensPerSecond" -> current.copy(tokensPerSecond = value as? Float ?: current.tokensPerSecond)
                "topWords" -> current.copy(topWords = (value as? List<*>)?.filterIsInstance<String>() ?: current.topWords)
                else -> current
            }
            updated
        }
    }

    private fun _infer(inputText: String): String {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return "[ERRO]"
        }
        val inputsMap = getTokenizeInput(inputText)
        val outputShape = localInterpreter.getOutputTensor(0).shape()
        val outBuffer: FloatBuffer = FloatBuffer.allocate(outputShape[1]+(201028-9220))
        val outputsMap = mapOf("logits" to outBuffer)


        val startTime = SystemClock.uptimeMillis()
        localInterpreter.runSignature(inputsMap, outputsMap, "infer")
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        outBuffer.rewind()
        val logitsArray = FloatArray(outBuffer.capacity())
        outBuffer.get(logitsArray)

        val k = 5
        val topKList = topK(logitsArray, k)

        _tokensPerSecond = 5/(inferenceTime.toFloat()/1000)

        val topKDecoded = topKList.map { (idx, prob) ->
            val tokenStr = tokenizer!!.decode(listOf(idx)) ?: "[UNK]"
            tokenStr to prob
        }
        updateModelMetric("tokensPerSecond", _tokensPerSecond)
        updateModelMetric("inferenceTime", inferenceTime)

        topKDecoded.forEachIndexed { rank, (tokenStr, p) ->
            Log.i(TAG, "   #${rank+1}: '$tokenStr' (prob=%.4f)".format(p))
        }

        val topWords = topKDecoded.map {it.first}
        updateModelMetric("topWords", topWords)

        val bestToken = topKDecoded.firstOrNull()?.first ?: "[UNK]"
        return bestToken
    }

    private fun mock(text: String): Triple<Array<IntArray>, Array<IntArray>, Array<IntArray>> {
        val tokens = tokenizer?.encode(text)?: throw IllegalStateException("Tokenização falhou.")
        return nextWordGenerate(tokens)
    }

    suspend fun train(inputText: String) {
        return withContext(Dispatchers.IO) {
            var peakUsageBytes = 0L
            var monitoring = true
            var runtime = Runtime.getRuntime()

            val monitorThread = thread(start = true) {
                while (monitoring) {
                    val jvmHeap = runtime.totalMemory() - runtime.freeMemory()
                    val totalUsage = jvmHeap + _initMem
                    peakUsageBytes = max(peakUsageBytes, totalUsage)
                    try {
                        Thread.sleep(50)
                    } catch (e: InterruptedException) {
                        break
                    }
                }
            }
            val trainingThread = thread(start = true) {
                _train(inputText)
                monitoring = false
            }

            monitorThread.join()
            trainingThread.join()

            val peakUsageMb = peakUsageBytes / (1024 * 1024)
            _memoryUsage.tryEmit(Pair(peakUsageMb, _totalMemGb))
        }
    }

    fun nextWordGenerate(vetor: MutableList<Int>, size: Int = 8): Triple<Array<IntArray>, Array<IntArray>, Array<IntArray>> {
        val samples = mutableListOf<IntArray>()
        val nextWords = mutableListOf<IntArray>()
        val lastTokens = if (vetor.size < size) vetor else vetor.takeLast(size)
        for (i in 1 until lastTokens.size) {
            val input = MutableList<Int>(size) { 0 }
            val label = MutableList<Int>(size) { 0 }
            for (j in 0 until size) {
                if (j >= i) { break; }
                input[i-j-1] = lastTokens[i-j-1]
                label[i-j-1] = lastTokens[i-j-1]
                label[i-j] = lastTokens[i-j]
            }
            nextWords.add(label.toIntArray())
            samples.add(input.toIntArray())
        }

        val mask = List<IntArray>(lastTokens.size-1) {
            List<Int>(size) { 0 }.toIntArray()
        }

        for (i in 0 until samples.size) {
            for (j in 0 until  samples[i].size) {
                if (samples[i][j] != 0) {
                    mask[i][j] = 1;
                }
            }
        }

        return Triple(samples.toTypedArray(), mask.toTypedArray(), nextWords.toTypedArray())
    }

    private fun _train(inputText: String) {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return
        }

        val (inputs, mask, labels) = mock(inputText)
        val outputShape = localInterpreter.getOutputTensor(0).shape()
        val outputBuffer: FloatBuffer = FloatBuffer.allocate(outputShape[1]*inputs.size)
        val outputs = mutableMapOf<String, Any>()
        outputs["loss"] = outputBuffer
        val losses = mutableListOf<Float>()

        val startTime = SystemClock.uptimeMillis()
        for (epoch in 0..10) {
            val inputsMap = mapOf(
                "input_ids" to inputs,
                "attention_mask" to mask,
                "labels" to labels
            )
            localInterpreter.runSignature(inputsMap, outputs, "train")
            outputBuffer.rewind()
            val current_loss = outputBuffer.get()
            losses.add(current_loss)
            outputBuffer.rewind()
            updateModelMetric("loss", current_loss)
            sleep(2000)
        }

        val trainingTime = SystemClock.uptimeMillis() - startTime
        val outputLoss: Float = outputBuffer.get()

        Log.i(TAG, "Treinamento finalizado em ${trainingTime}ms")

        outputBuffer.rewind()
        updateModelMetric("loss", outputLoss)
        updateModelMetric("tokensPerSecond", _tokensPerSecond)
        updateModelMetric("trainingTime", trainingTime)
        return
    }

    suspend fun textChange(inputText: String) {
        withContext(Dispatchers.IO) {
            _textState.value = inputText
            return@withContext
        }
    }

    private fun topK(probs: FloatArray, k: Int = 5): List<Pair<Int, Float>> {
        val logitsArray = probs.take(50256)
        val indexed = logitsArray.mapIndexed { idx, p -> idx to p }
        return indexed.sortedByDescending { it.second }.take(k)
    }

    private fun loadEncoder(context: Context): Map<String, Int> {
        val json = context.assets.open("gpt2/vocab.json").bufferedReader().use { it.readText() }
        return Gson().fromJson(json, object : TypeToken<Map<String, Int>>() {}.type)
    }

    private fun loadBpeRanks(context: Context): Map<Pair<String, String>, Int> {
        val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
        val lines = context.assets.open("gpt2/merges.txt").bufferedReader().readLines()
        var rank = 0
        for (line in lines) {
            if (line.startsWith("#")) continue
            val tokens = line.split(" ")
            if (tokens.size == 2) {
                bpeRanks[Pair(tokens[0], tokens[1])] = rank
                rank++
            }
        }
        return bpeRanks
    }
    companion object {
        private const val TAG = "TextClassifier"
    }

    enum class Model(val fileName: String) {
        LocalModel("model.tflite"),
        LoraModel("lora_model.tflite"),
    }
}
