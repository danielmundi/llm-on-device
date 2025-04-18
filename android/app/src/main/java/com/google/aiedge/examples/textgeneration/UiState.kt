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

import androidx.compose.runtime.Immutable

data class ModelInfo(
    var inferenceTime: Long = 0L,
    var trainingTime: Long = 0L,
    var loss: Float = 0f,
    var tokensPerSecond: Float = 0f,
    var topWords: List<String> = listOf<String>(),
)

@Immutable
data class UiState(
    val errorMessage: String? = null,
    val modelInfo: ModelInfo = ModelInfo(),
    val textTyped: String = "",
    val memoryUsage: Pair<Long, Long> = Pair(0L, 0L),
    val weightSelected: String = "",
)