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

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.WhileSubscribed
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val textGenerationHelper: TextGenerationHelper) : ViewModel() {
    companion object {
        fun getFactory(context: Context) = object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val textGenerationHelper = TextGenerationHelper(context)
                return MainViewModel(textGenerationHelper) as T
            }
        }
    }

    init {
        viewModelScope.launch {
            textGenerationHelper.initClassifier()
        }
    }

    private var generateJob: Job? = null
    private var trainingJob: Job? = null
    private var saveWeight: Job? = null
    private var typingJob: Job? = null
    private var restoreJob: Job? = null

    private val modelInfo = textGenerationHelper.modelInfo.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), ModelInfo()
    )

    private val memoryUsage = textGenerationHelper.memoryUsage.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), Pair(0L, 0L)
    )

    private val textTyped = textGenerationHelper.textState.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), ""
    )

    private val errorMessage = MutableStateFlow<Throwable?>(null).also {
        viewModelScope.launch {
            textGenerationHelper.error.collect(it)
        }
    }

    val uiState: StateFlow<UiState> = combine(
        errorMessage, modelInfo, textTyped, memoryUsage
    ) { throwable, modelInfo, textTyped, memoryUsage ->
        textGenerationHelper.completableDeferred?.complete(Unit)
        UiState(
            errorMessage = throwable?.message,
            modelInfo = modelInfo,
            textTyped = textTyped,
            memoryUsage = memoryUsage,
        )
    }.stateIn(
        viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState()
    )

    fun runClassification(inputText: String) {
        generateJob?.cancel()
        generateJob = viewModelScope.launch {
            textGenerationHelper.infer(inputText)
        }
    }
    fun runSaveW() {
        saveWeight?.cancel()
        saveWeight = viewModelScope.launch {
            textGenerationHelper.save("trained_model")
        }
    }

    fun runRestore(fileName: String) {
        restoreJob?.cancel()
        restoreJob = viewModelScope.launch {
            textGenerationHelper.restore(fileName)
        }
    }

    fun runTraining(inputText: String) {
        trainingJob?.cancel()
        trainingJob = viewModelScope.launch {
            textGenerationHelper.train(inputText)
        }
    }
    fun textChange(inputText: String) {
        typingJob?.cancel()
        typingJob = viewModelScope.launch {
            textGenerationHelper.textChange(inputText)
        }
    }

    fun setModel(model: TextGenerationHelper.Model) {
        viewModelScope.launch {
            textGenerationHelper.stopClassify()
            textGenerationHelper.initClassifier(model)
        }
    }

    fun errorMessageShown() {
        errorMessage.update { null }
    }
}