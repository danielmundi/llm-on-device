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

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.aiedge.examples.textgeneration.ui.ApplicationTheme
import com.google.aiedge.examples.textgeneration.ui.mainOrange

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TextGenerationScreen()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextGenerationScreen(
    viewModel: MainViewModel = viewModel(
        factory = MainViewModel.getFactory(LocalContext.current.applicationContext)
    )
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(key1 = uiState.errorMessage) {
        if (uiState.errorMessage != null) {
            Toast.makeText(
                context,
                "${uiState.errorMessage}",
                Toast.LENGTH_SHORT
            ).show()
            viewModel.errorMessageShown()
        }
    }

    ApplicationTheme {
        BottomSheetScaffold(
            sheetDragHandle = {
                Image(
                    modifier = Modifier
                        .size(40.dp)
                        .padding(top = 2.dp, bottom = 5.dp),
                    painter = painterResource(id = R.drawable.ic_chevron_up),
                    colorFilter = ColorFilter.tint(mainOrange),
                    contentDescription = ""
                )
            },
            sheetPeekHeight = 70.dp,
            topBar = {
                Header()
            },
            sheetContent = {
                BottomSheetContent(
                    inferenceTime = uiState.modelInfo.inferenceTime,
                    trainingTime = uiState.modelInfo.trainingTime,
                    onModelSelected = {
                        viewModel.setModel(it)
                    },
                )
            }) {
            GenerationBody(
                trainingLoss = uiState.modelInfo.loss,
                textTyped = uiState.textTyped,
                onSubmittedGenerate = {
                    if (it.isNotBlank()) {
                        viewModel.runClassification(it)
                    }
                },
                onTextChange = {
                    viewModel.textChange(it)
                },
                onSubmittedTrain = {
                    if (it.isNotBlank()) {
                        viewModel.runTraining(it)
                    }
                },
                onSubmittedSaveW = { viewModel.runSaveW() },
                onSubmittedRestore = { viewModel.runRestore(it) },
                memoryUsage = uiState.memoryUsage,
                tokensPerSecond = uiState.modelInfo.tokensPerSecond,
                topWords = uiState.modelInfo.topWords,
                onWordSelected = {
                    viewModel.textChange(it)
                }
            )

        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Header() {
    TopAppBar(
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = Color.White
        ),
        title = {
            Image(
                modifier = Modifier.size(120.dp),
                alignment = Alignment.CenterStart,
                painter = painterResource(id = R.drawable.logo),
                contentDescription = null,
            )
        },
    )
}

@Composable
fun BottomSheetContent(
    inferenceTime: Long,
    trainingTime: Long,
    modifier: Modifier = Modifier,
    onModelSelected: (TextGenerationHelper.Model) -> Unit,
) {
    Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 5.dp)) {
        Row {
            Text(
                modifier = Modifier.weight(0.5f),
                text = stringResource(id = R.string.inference_title),
                fontSize = 16.sp
            )
            Text(
                text = stringResource(id = R.string.inference_value, inferenceTime),
                fontSize = 16.sp
            )
        }
        Row{
            Text(
                modifier = Modifier.weight(0.5f),
                text = stringResource(id = R.string.training_title),
                fontSize = 16.sp
            )
            Text(
                text = stringResource(id = R.string.training_value, trainingTime),
                fontSize = 16.sp
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        ModelSelection(
            onModelSelected = onModelSelected,
        )
    }
}

@Composable
fun GenerationBody(
    trainingLoss: Float,
    textTyped: String,
    modifier: Modifier = Modifier,
    onSubmittedGenerate: (String) -> Unit,
    onTextChange: (String) -> Unit,
    onSubmittedTrain: (String) -> Unit,
    onSubmittedSaveW: () -> Unit,
    onSubmittedRestore: (String) -> Unit,
    memoryUsage: Pair<Long, Long>,
    tokensPerSecond: Float,
    topWords: List<String>,
    onWordSelected: (String) -> Unit,
) {
    val focusManager = LocalFocusManager.current

    Column(
        modifier = modifier.padding(horizontal = 20.dp),
    ) {
        Spacer(modifier = Modifier.height(20.dp))
        TopKSelector(words = topWords, onWordSelected = {
                 onWordSelected("${textTyped}${it}")
        },)
        TextField(
            modifier = Modifier
                .fillMaxWidth()
                .height(100.dp),
            value = textTyped,
            onValueChange = onTextChange,
            placeholder = {
                Text(text = stringResource(id = R.string.text_field_place_holder))
            },

        )
        Spacer(modifier = Modifier.height(10.dp))
        RestoreWeightsSelector(
            onRestoreWeights = onSubmittedRestore
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ){
            Button(
                onClick = {
                    focusManager.clearFocus()
                    onSubmittedGenerate(textTyped)
                }) {
                Text(text = stringResource(id = R.string.generate))
            }
            Spacer(modifier = Modifier.width(20.dp))
            Button(
                onClick = {
                    focusManager.clearFocus()
                    onSubmittedTrain(textTyped)
                }) {
                Text(text = stringResource(id = R.string.train))
            }
            Spacer(modifier = Modifier.width(20.dp))
            Button(
                onClick = {
                    focusManager.clearFocus()
                    onSubmittedSaveW()
                }) {
                Text(text = stringResource(id = R.string.save_w))
            }
            Spacer(modifier = Modifier.width(20.dp))
            Button(
                onClick = {
                    focusManager.clearFocus()
                    onSubmittedRestore("default_lora_weights")
                }) {
                Text( text = stringResource(id = R.string.restore))
            }
        }
        Spacer(modifier = Modifier.height(20.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = stringResource(id = R.string.training_loss, trainingLoss),
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.width(20.dp))
            Text(
                text = stringResource(id = R.string.tokens_per_second, tokensPerSecond),
            )
        }
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = stringResource(id = R.string.memory, memoryUsage.first, memoryUsage.second),
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun ModelSelection(
    modifier: Modifier = Modifier,
    onModelSelected: (TextGenerationHelper.Model) -> Unit,
) {
    val radioOptions = TextGenerationHelper.Model.entries.map { it.name }.toList()
    var selectedOption by remember { mutableStateOf(radioOptions.first()) }

    Column(modifier = modifier) {
        radioOptions.forEach { option ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(
                    selected = (option == selectedOption),
                    onClick = {
                        if (selectedOption == option) return@RadioButton
                        onModelSelected(TextGenerationHelper.Model.valueOf(option))
                        selectedOption = option
                    },
                )
                Text(modifier = Modifier.padding(start = 16.dp), text = option, fontSize = 15.sp)
            }
        }
    }
}

@Composable
fun TopKSelector(
    words: List<String>,
    onWordSelected: (String) -> Unit,
) {
    var selectedWord by remember { mutableStateOf<String?>(null) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        words.forEach { word ->
            Text(
                text = word,
                fontSize = 18.sp,
                color = if (word == selectedWord) Color.Red else Color.Black,
                modifier = Modifier
                    .clickable {
                        selectedWord = word
                        onWordSelected(word)
                    }
                    .padding(8.dp)
            )
        }
    }
}

@Composable
fun RestoreWeightsSelector(
    onRestoreWeights: (filePath: String) -> Unit
) {
    val options = listOf(
        "Fine Tuned 1" to "finetuned1",
        "Fine Tuned 2" to "finetuned2",
        "Fine Tuned 3" to "finetuned3"
    )
    var selectedOption by remember { mutableStateOf(options.first().first) }

    Column {
        options.forEach { (label, filePath) ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        if (selectedOption != label) {
                            selectedOption = label
                            onRestoreWeights(filePath)
                        }
                    }
                    .padding(4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                RadioButton(
                    selected = (selectedOption == label),
                    onClick = null
                )
                Text(
                    text = label,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(start = 4.dp)
                )
            }
        }
    }
}
