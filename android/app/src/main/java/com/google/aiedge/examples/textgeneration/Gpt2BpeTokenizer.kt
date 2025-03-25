/**
 * This code is based on the implementation found in the following repository:
 * https://github.com/huggingface/tflite-android-transformers
 *
 * The referenced implementation provides functionality for tokenizing text
 * using the GPT-2 model's tokenization logic.
 */
package com.google.aiedge.examples.textgeneration

// Classe GPT2Tokenizer, responsável por codificar e decodificar texto usando o modelo GPT-2
class GPT2Tokenizer(
    private val encoder: Map<String, Int>,  // Mapeamento de tokens para ids numéricos (codificação)
    private val decoder: Map<Int, String>,  // Mapeamento de ids numéricos para tokens (decodificação)
    private val bpeRanks: Map<Pair<String, String>, Int>  // Mapeamento de pares de tokens (sub-palavras) para rankings
) {
    // Expressão regular para segmentação de texto (tokenização)
    private val encodeRegex = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    // Função para decodificar uma lista de tokens (ids) de volta para o texto
    fun decode(tokens: List<Int>): String {
        // Mapeia os tokens para suas representações em string e os junta
        val text = tokens.joinToString("") { decoder.getOrDefault(it, "") }
        // Mapeia os caracteres de volta para seus códigos UTF-8 e cria uma string com eles
        val utfCodepoints = text.map { byteDecoder[it.toString()]!! }
        return String(utfCodepoints.toIntArray(), 0, utfCodepoints.size)
    }

    // Função para codificar um texto (string) em uma lista de tokens (ids)
    fun encode(text: String): MutableList<Int> {
        // Utiliza a regex para dividir o texto em partes significativas (tokens)
        val tokens = encodeRegex.findAll(text).map { result ->
            result.value.codePoints()
                .boxed()  // Converte os pontos de código para uma lista
                .map { byteEncoder[it]!! }  // Mapeia cada ponto de código para seu id correspondente
                .toArray()
                .joinToString("")  // Junta os ids dos caracteres em uma string
        }

        // Aplica a codificação Byte Pair Encoding (BPE) e converte cada token em seu id correspondente
        return tokens
            .map { bpe(it) }  // Aplica BPE em cada token
            .flatten()  // Flatten transforma a lista de listas em uma lista simples
            .map { encoder[it]!! }  // Converte cada sub-token em seu id numérico
            .toMutableList()
    }

    // Função de codificação Byte Pair Encoding (BPE) para segmentação de palavras
    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)  // Se o token tem apenas 1 caractere, não aplica BPE

        var word = token.map { it.toString() }  // Divide a palavra em caracteres individuais
        var pairs = getPairs(word)  // Obtém os pares de caracteres contíguos

        // Continua a aplicar a BPE enquanto houver pares que possam ser mesclados
        while (true) {
            if (!pairs.any { bpeRanks.containsKey(it) }) break  // Se não houver pares válidos, interrompe
            val (first, second) = pairs.minBy { bpeRanks.getOrDefault(it, Int.MAX_VALUE) } ?: break

            var i = 0
            val newWord = mutableListOf<String>()
            // Percorre a palavra buscando os pares e os substitui por uma única unidade
            while (i < word.size) {
                val j = word.withIndex().indexOfFirst { it.index >= i && it.value == first }
                if (j != -1) {
                    newWord.addAll(word.subList(i, j))  // Adiciona a parte anterior ao novo token
                    i = j
                } else {
                    newWord.addAll(word.subList(i, word.size))  // Se não encontrar o par, adiciona o restante
                    break
                }

                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)  // Substitui o par por um único token
                    i += 2
                } else {
                    newWord.add(word[i])  // Caso contrário, mantém o token individual
                    i += 1
                }
            }

            word = newWord  // Atualiza a palavra com os novos tokens
            if (word.size == 1) {
                break  // Se a palavra for reduzida a um único token, interrompe
            } else {
                pairs = getPairs(word)  // Atualiza os pares para o próximo ciclo de BPE
            }
        }

        return word  // Retorna a lista de sub-tokens
    }

    // Função para obter todos os pares consecutivos de caracteres (tokens)
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return mutableSetOf<Pair<String, String>>().apply {
            // Adiciona todos os pares consecutivos de tokens
            for (i in 0 until word.size - 1) {
                add(word[i] to word[i + 1])
            }
        }
    }
}
