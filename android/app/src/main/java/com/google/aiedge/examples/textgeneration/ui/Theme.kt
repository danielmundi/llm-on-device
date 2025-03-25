package com.google.aiedge.examples.textgeneration.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val colorScheme = lightColorScheme(
    primary = darkPurple,
    secondary = textPrimary,
    onSurfaceVariant = textPrimary,
    background = Color.White,
    onBackground = Color.White
)

@Composable
fun ApplicationTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
