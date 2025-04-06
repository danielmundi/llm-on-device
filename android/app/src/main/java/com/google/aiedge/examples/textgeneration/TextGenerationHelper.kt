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
import android.os.SystemClock.sleep
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.nio.FloatBuffer
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

        initClassifier(Model.LoraModel)
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

    private val _weightSelected = MutableStateFlow("finetuned1")
    val weightSelected: StateFlow<String> get() = _weightSelected

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    var completableDeferred: CompletableDeferred<Unit>? = null

    fun initClassifier(model: Model = Model.LoraModel) {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(info)
        _totalMemGb = (info.totalMem / (1024 * 1024 * 1024))+1

        interpreter = try {
            val tfliteBuffer = FileUtil.loadMappedFile(context, model.fileName)
            _initMem = tfliteBuffer.capacity().toLong()
            Log.i(TAG, "LiteRT buffer criado a partir de ${model.fileName}")
            // Leitura do modelo .tflite
            Interpreter(tfliteBuffer, Interpreter.Options())

        } catch (e: Exception) {
            Log.e(TAG, "Falha ao criar LiteRT a partir de ${model.fileName}: ${e.message}", e)
            null
        }
        if (model.fileName == "lora_model.tflite") {
            save("default_lora_weights")
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

    /**
     * Realiza a inferência do modelo com base no texto de entrada fornecido e retorna o token mais provável.
     *
     * @param inputText Texto de entrada que será analisado pelo modelo.
     * @return O token mais provável após a inferência do modelo.
     */
    private fun _infer(inputText: String): String {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return "[ERRO]"
        }

        // Prepara o mapa de entradas tokenizadas para o modelo.
        val inputsMap = getTokenizeInput(inputText)

        // Obtém a forma (shape) da saída do tensor (output) do modelo para o índice 0.
        val outputShape = localInterpreter.getOutputTensor(0).shape()

        // Cria um buffer para armazenar os resultados da saída, com base na forma da saída.
        val outBuffer: FloatBuffer = FloatBuffer.allocate(outputShape[1] + (201028 - 9220))

        // Cria um mapa de saídas, associando a chave "logits" ao buffer de saída.
        val outputsMap = mapOf("logits" to outBuffer)

        val startTime = SystemClock.uptimeMillis()

        // Executa a inferência no modelo, passando as entradas e as saídas.
        // O método runSignature é usado para executar a inferência com a assinatura especificada.
        localInterpreter.runSignature(inputsMap, outputsMap, "infer")

        val inferenceTime = SystemClock.uptimeMillis() - startTime

        // Rewind do buffer de saída para garantir que ele está pronto para leitura.
        outBuffer.rewind()

        // Cria um array para armazenar os valores dos logits (saídas do modelo).
        val logitsArray = FloatArray(outBuffer.capacity())

        // Preenche o array de logits a partir do buffer.
        outBuffer.get(logitsArray)

        // Define k como o número de tokens mais prováveis que queremos retornar.
        val k = 5

        // Obtém os k tokens mais prováveis (top-k) a partir do array de logits.
        val topKList = topK(logitsArray, k)

        // Calcula a taxa de tokens por segundo com base no tempo de inferência.
        _tokensPerSecond = 5 / (inferenceTime.toFloat() / 1000)

        // Decodifica os tokens mais prováveis e os mapeia para uma lista de pares (token, probabilidade).
        val topKDecoded = topKList.map { (idx, prob) ->
            val tokenStr = tokenizer!!.decode(listOf(idx)) ?: "[UNK]" // Decodifica o token usando o tokenizer, ou retorna "[UNK]" caso o token não seja encontrado.
            tokenStr to prob
        }

        updateModelMetric("tokensPerSecond", _tokensPerSecond)
        updateModelMetric("inferenceTime", inferenceTime)

        val topWords = topKDecoded.map { it.first }

        updateModelMetric("topWords", topWords)

        val bestToken = topKDecoded.firstOrNull()?.first ?: "[UNK]" // Se não houver nenhum token, retorna "[UNK]".
        return bestToken
    }

//    private fun mock(text: String): Triple<Array<IntArray>, Array<IntArray>, Array<IntArray>> {
//        val tokens = tokenizer?.encode(text)?: throw IllegalStateException("Tokenização falhou.")
//        return nextWordGenerate(tokens)
//    }

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

//    fun nextWordGenerate(vetor: MutableList<Int>, size: Int = 8): Triple<Array<IntArray>, Array<IntArray>, Array<IntArray>> {
//        val samples = mutableListOf<IntArray>()
//        val nextWords = mutableListOf<IntArray>()
//        val lastTokens = if (vetor.size < size) vetor else vetor.takeLast(size)
//        for (i in 1 until lastTokens.size) {
//            val input = MutableList<Int>(size) { 0 }
//            val label = MutableList<Int>(size) { 0 }
//            for (j in 0 until size) {
//                if (j >= i) { break; }
//                input[i-j-1] = lastTokens[i-j-1]
//                label[i-j-1] = lastTokens[i-j-1]
//                label[i-j] = lastTokens[i-j]
//            }
//            nextWords.add(label.toIntArray())
//            samples.add(input.toIntArray())
//        }
//
//        val mask = List<IntArray>(lastTokens.size-1) {
//            List<Int>(size) { 0 }.toIntArray()
//        }
//
//        for (i in 0 until samples.size) {
//            for (j in 0 until  samples[i].size) {
//                if (samples[i][j] != 0) {
//                    mask[i][j] = 1;
//                }
//            }
//        }
//
//        return Triple(samples.toTypedArray(), mask.toTypedArray(), nextWords.toTypedArray())
//    }

    /*
    "my name is|Daniel#my name is|Daniel#my name is|Daniel#my name is|Daniel"
    "my name is|Daniel".split("|").join(" ")
    my name is Daniel
     */

    private fun prepareInputLabel(inputSample: String, labelSample: String, size: Int = 8): Triple<Array<IntArray>, Array<IntArray>, Array<IntArray>> {
        Log.i(TAG, "inputSample = $inputSample")
        Log.i(TAG, "labelSample = $labelSample")
        val inputTokens = tokenizer?.encode(inputSample)?: throw IllegalStateException("Tokenização falhou.")
        val labelTokens = tokenizer?.encode(labelSample)?: throw IllegalStateException("Tokenização falhou.")

        val samplesTrain = mutableListOf<IntArray>()
        val nextWordsTrain = mutableListOf<IntArray>()
        val trainTokens = if (inputTokens.size < size) inputTokens else inputTokens.takeLast(size)

        val inputTrain = MutableList<Int>(size) { 0 }
        val labelTrain = MutableList<Int>(size) { 0 }

        var i_input = 0
        var i_label = 0

        for (tk in trainTokens) {
            inputTrain[i_input] = tk
            labelTrain[i_input] = tk
            i_input++
        }
        //inputTrain.addAll(trainTokens.toTypedArray())
        for (label in labelTokens) {
            // Add each token to the array to be trained
            labelTrain[i_input] = label

            Log.i(TAG, "fill[$i_input] inputTrain = " + inputTrain.joinToString(separator = ", ", prefix = "[", postfix = "]"))
            samplesTrain.add(inputTrain.toIntArray())
            Log.i(TAG, "fill[$i_input] labelTrain = " + labelTrain.joinToString(separator = ", ", prefix = "[", postfix = "]"))
            nextWordsTrain.add(labelTrain.toIntArray())

            // Add the same label to the samples in case there are more than one label
            inputTrain[i_input] = label
            i_input++
        }

        Log.i(TAG, "Samples size = " + samplesTrain.size)
        val maskTrain = List<IntArray>(samplesTrain.size) {
            List<Int>(size) { 0 }.toIntArray()
        }
        for (i in 0 until samplesTrain.size) {
            for (j in 0 until  samplesTrain[i].size) {
                Log.i(TAG, "Process maskTran[$i][$j]")
                if (samplesTrain[i][j] != 0) {
                    Log.i(TAG, "Set maskTran[$i][$j] = 1")
                    maskTrain[i][j] = 1;
                }
            }
        }

        for (i in 0 until(samplesTrain.size)) {
            Log.i(TAG, "[$i] samplesTrain = " + samplesTrain[i].joinToString(separator = ", ", prefix = "[", postfix = "]"))
            Log.i(TAG, "[$i] maskTrain = " + maskTrain[i].joinToString(separator = ", ", prefix = "[", postfix = "]"))
            Log.i(TAG, "[$i] nextWordsTrain = " + nextWordsTrain[i].joinToString(separator = ", ", prefix = "[", postfix = "]"))
        }

        return Triple(samplesTrain.toTypedArray(), maskTrain.toTypedArray(), nextWordsTrain.toTypedArray())
    }

    /**
     * Realiza o treinamento do modelo utilizando o texto de entrada fornecido.
     * Durante o treinamento, o modelo é alimentado com entradas e etiquetas e o erro (loss) é computado a cada época.
     *
     * @param inputText Texto de entrada que será utilizado no treinamento.
     */
    private fun _train(inputText: String) {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return
        }

        Log.i(TAG, "Input text = $inputText")

        var inputsM = mutableListOf<IntArray>()
        var maskM = mutableListOf<IntArray>()
        var labelsM = mutableListOf<IntArray>()

        val inputSamples = inputText.split("#").toTypedArray()
        for (sample in inputSamples) {
            val (inputSample, labelSample) = sample.split("|", limit=2)
            val (inputs_sample, mask_sample, labels_sample) = prepareInputLabel(inputSample, labelSample)
            inputsM.addAll(inputs_sample)
            maskM.addAll(mask_sample)
            labelsM.addAll(labels_sample)
        }

        var inputs = inputsM.toTypedArray()
        var mask = maskM.toTypedArray()
        var labels = labelsM.toTypedArray()

        // Gera dados simulados (mock) a partir do texto de entrada.
        // 'inputs', 'mask' e 'labels' são extraídos do mock.
        //val (inputs, mask, labels) = mock(inputText)

        // Obtém a forma (shape) da saída do tensor (output) do modelo para o índice 0.
        val outputShape = localInterpreter.getOutputTensor(0).shape()

        // Cria um buffer para armazenar os valores de perda (loss) durante o treinamento.
        val outputBuffer: FloatBuffer = FloatBuffer.allocate(outputShape[1] * inputs.size)

        // Cria um mapa de saídas e associa a chave "loss" ao buffer de saída.
        val outputs = mutableMapOf<String, Any>()
        outputs["loss"] = outputBuffer

        // Lista para armazenar as perdas (losses) durante o treinamento.
        val losses = mutableListOf<Float>()

        // Marca o início do tempo para medir a duração total do treinamento.
        val startTime = SystemClock.uptimeMillis()

        // Executa o treinamento por 10 épocas
        for (epoch in 0..40) {
            Log.i(TAG, "Train epoch = " + epoch)
            // Cria o mapa de entradas a ser alimentado ao modelo.
            val inputsMap = mapOf(
                "input_ids" to inputs, // Identificadores de entrada (tokens)
                "attention_mask" to mask, // Máscara de atenção para indicar quais tokens são relevantes
                "labels" to labels // targets
            )

            localInterpreter.runSignature(inputsMap, outputs, "train")

            // Rewind do buffer de saída para garantir que ele está pronto para leitura.
            outputBuffer.rewind()

            // Obtém o valor da perda (loss) calculada pelo modelo para esta época.
            val current_loss = outputBuffer.get()
            losses.add(current_loss)

            // Rewind novamente o buffer para leitura.
            outputBuffer.rewind()
            updateModelMetric("loss", current_loss)
            if (epoch > 30 && current_loss < 1.2) {
                break
            }
            // Faz uma pausa de 1 segundo entre as épocas para previnir sobrecarga no dispositivo
            sleep(500)
        }
        Log.i(TAG, "Train done")

        val trainingTime = SystemClock.uptimeMillis() - startTime

        // Obtém a última perda registrada após a última época de treinamento.
        val outputLoss: Float = outputBuffer.get()

        // Rewind do buffer de saída antes de atualizar as métricas.
        outputBuffer.rewind()

        updateModelMetric("loss", outputLoss)
        updateModelMetric("tokensPerSecond", _tokensPerSecond)
        updateModelMetric("trainingTime", trainingTime)

        return
    }

    fun textChange(inputText: String) {
        //withContext(Dispatchers.IO) {
            _textState.value = inputText
            //return@withContext
        //}
    }

    fun weightChange(newWeight: String) {
        //withContext(Dispatchers.IO) {
            _weightSelected.value = newWeight
            //return@withContext
        //}
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

    fun save(fileName: String) {
        _save(context, fileName)
    }

    private fun _save(context: Context, fileName: String): String   {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return "[ERRO]"
        }
        val outputFile = File(context.filesDir, fileName)

        val inputs: MutableMap<String, Any> = HashMap()
        inputs["checkpoint_path"] = outputFile.absolutePath
        val outputs: Map<String, Any> = HashMap()
        localInterpreter.runSignature(inputs, outputs, "save")
        if (outputFile.exists() && outputFile.length() > 0) {
            Log.d(TAG, "Modelo salvo com sucesso em: ${outputFile.absolutePath}")
            "Saved"
        } else {
            Log.e(TAG, "Falha ao salvar o modelo. O arquivo está vazio ou não foi criado.")
            "[ERRO] Arquivo não salvo corretamente."
        }
        return "Saved"
    }

    fun restore(fileName: String) {
        _restore(context, fileName)
    }

    private fun _restore(context: Context, fileName: String): String {
        val localInterpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter não inicializado.")
            return "[ERRO]"
        }
        val outputFile = File(context.filesDir, fileName)
        if (outputFile.exists() && outputFile.length() > 0) {
            val inputs: MutableMap<String, Any> = HashMap()
            inputs["checkpoint_path"] = outputFile.absolutePath
            val outputs: Map<String, Any> = HashMap()

            localInterpreter.runSignature(inputs, outputs, "restore")
            Log.d(TAG, "Modelo restaurado com sucesso em: ${outputFile.absolutePath}")
            return "Restore"
        }
        Log.e(TAG, "Falha ao restaurar o modelo. O arquivo está vazio ou não foi criado.")
        return "Not Restore"
    }

    companion object {
        private const val TAG = "TextClassifier"
    }

    enum class Model(val fileName: String) {
        //FullModel("model.tflite"),
        LoraModel("lora_model.tflite"),
    }
}
