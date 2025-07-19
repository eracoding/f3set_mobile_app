package com.example.aivideoanalyzer.presentation.ui.results

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.aivideoanalyzer.AIVideoAnalyzerApplication
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import com.example.aivideoanalyzer.domain.usecase.GenerateReportUseCase
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class ResultsViewModel(application: Application) : AndroidViewModel(application) {

    sealed class ExportState {
        object Idle : ExportState()
        object Exporting : ExportState()
        data class Success(val filePath: String) : ExportState()
        data class Error(val message: String) : ExportState()
    }

    sealed class ShareState {
        object Idle : ShareState()
        data class ReadyToShare(val shareIntent: Intent) : ShareState()
        data class ReadyToSave(val saveIntent: Intent) : ShareState()
        data class Error(val message: String) : ShareState()
    }

    enum class SortOption {
        DATE_NEWEST,
        DATE_OLDEST,
        CONFIDENCE_HIGH,
        CONFIDENCE_LOW
    }

    private val _analysisResults = MutableLiveData<List<AnalysisResult>>(emptyList())
    val analysisResults: LiveData<List<AnalysisResult>> = _analysisResults

    private val _selectedResult = MutableLiveData<AnalysisResult?>()
    val selectedResult: LiveData<AnalysisResult?> = _selectedResult

    private val _exportState = MutableLiveData<ExportState>(ExportState.Idle)
    val exportState: LiveData<ExportState> = _exportState

    private val _shareState = MutableLiveData<ShareState>(ShareState.Idle)
    val shareState: LiveData<ShareState> = _shareState

    private var allResults: List<AnalysisResult> = emptyList()
    private var currentSortOption = SortOption.DATE_NEWEST

    private val app = application as AIVideoAnalyzerApplication
    private val videoRepository = app.videoRepository
    private val generateReportUseCase = app.generateReportUseCase

    init {
        loadResults()
        observeResults()
    }

    private fun loadResults() {
        viewModelScope.launch {
            videoRepository.getAllAnalysisResults().fold(
                onSuccess = { results ->
                    allResults = results
                    _analysisResults.value = sortResults(results)
                },
                onFailure = {
                    // Handle error
                }
            )
        }
    }

    private fun observeResults() {
        viewModelScope.launch {
            videoRepository.observeAnalysisResults().collect { results ->
                allResults = results
                _analysisResults.value = sortResults(results)
            }
        }
    }

    fun searchResults(query: String) {
        viewModelScope.launch {
            if (query.isBlank()) {
                _analysisResults.value = sortResults(allResults)
            } else {
                val filtered = allResults.filter { result ->
                    result.videoId.contains(query, ignoreCase = true) ||
                            result.summary.contains(query, ignoreCase = true)
                }
                _analysisResults.value = sortResults(filtered)
            }
        }
    }

    fun sortResults(option: Int) {
        currentSortOption = when (option) {
            0 -> SortOption.DATE_NEWEST
            1 -> SortOption.DATE_OLDEST
            2 -> SortOption.CONFIDENCE_HIGH
            3 -> SortOption.CONFIDENCE_LOW
            else -> SortOption.DATE_NEWEST
        }

        _analysisResults.value = sortResults(_analysisResults.value ?: emptyList())
    }

    private fun sortResults(results: List<AnalysisResult>): List<AnalysisResult> {
        return when (currentSortOption) {
            SortOption.DATE_NEWEST -> results.sortedByDescending { it.timestamp }
            SortOption.DATE_OLDEST -> results.sortedBy { it.timestamp }
            SortOption.CONFIDENCE_HIGH -> results.sortedByDescending { it.confidence }
            SortOption.CONFIDENCE_LOW -> results.sortedBy { it.confidence }
        }
    }

    fun refreshResults() {
        loadResults()
    }

    /**
     * Export result and prepare for sharing/saving
     */
    fun exportResult(result: AnalysisResult, format: String) {
        viewModelScope.launch {
            _exportState.value = ExportState.Exporting

            try {
                val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
                val timestamp = dateFormat.format(Date())
                val fileName = "f3set_analysis_${result.videoId}_${timestamp}.$format"

                // Create file in app's cache directory first
                val tempFile = File(app.cacheDir, fileName)

                val exportResult = when (format) {
                    "html" -> generateReportUseCase.exportToExcel(result, tempFile.absolutePath)
                    "csv" -> generateReportUseCase.exportToCsv(result, tempFile.absolutePath)
                    "json" -> generateReportUseCase.exportToPdf(result, tempFile.absolutePath)
                    else -> Result.failure(Exception("Unknown format"))
                }

                exportResult.fold(
                    onSuccess = { filePath ->
                        _exportState.value = ExportState.Success(filePath)

                        // Prepare sharing intent
                        val file = File(filePath)
                        if (file.exists()) {
                            prepareShareIntent(file, format, result)
                        } else {
                            _shareState.value = ShareState.Error("Exported file not found")
                        }
                    },
                    onFailure = { error ->
                        _exportState.value = ExportState.Error(error.message ?: "Export failed")
                    }
                )

            } catch (e: Exception) {
                _exportState.value = ExportState.Error(e.message ?: "Export failed")
            }
        }
    }

    /**
     * Prepare sharing intent for exported file
     */
    private fun prepareShareIntent(file: File, format: String, result: AnalysisResult) {
        try {
            val uri = FileProvider.getUriForFile(
                app,
                "${app.packageName}.fileprovider",
                file
            )

            val mimeType = when (format) {
                "html" -> "text/html"
                "csv" -> "text/csv"
                "json" -> "application/json"
                else -> "application/octet-stream"
            }

            val formatName = when (format) {
                "html" -> "HTML Report"
                "csv" -> "CSV Data"
                "json" -> "JSON Data"
                else -> "Analysis File"
            }

            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = mimeType
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "🎾 F3Set Tennis Analysis - ${result.videoId}")
                putExtra(Intent.EXTRA_TEXT, buildShareText(result, formatName))
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            _shareState.value = ShareState.ReadyToShare(shareIntent)

        } catch (e: Exception) {
            _shareState.value = ShareState.Error("Failed to prepare sharing: ${e.message}")
        }
    }

    /**
     * Prepare save intent for exported file
     */
    fun prepareSaveIntent(result: AnalysisResult, format: String) {
        try {
            val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
            val timestamp = dateFormat.format(Date())
            val fileName = "f3set_analysis_${result.videoId}_${timestamp}.$format"

            val mimeType = when (format) {
                "html" -> "text/html"
                "csv" -> "text/csv"
                "json" -> "application/json"
                else -> "application/octet-stream"
            }

            val saveIntent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = mimeType
                putExtra(Intent.EXTRA_TITLE, fileName)
            }

            _shareState.value = ShareState.ReadyToSave(saveIntent)

        } catch (e: Exception) {
            _shareState.value = ShareState.Error("Failed to prepare save dialog: ${e.message}")
        }
    }

    /**
     * Save exported file to user-selected location
     */
    fun saveFileToUri(result: AnalysisResult, format: String, destinationUri: Uri) {
        viewModelScope.launch {
            _exportState.value = ExportState.Exporting

            try {
                // Create temporary file first
                val tempFile = File.createTempFile("f3set_export", ".$format", app.cacheDir)

                val exportResult = when (format) {
                    "html" -> generateReportUseCase.exportToExcel(result, tempFile.absolutePath)
                    "csv" -> generateReportUseCase.exportToCsv(result, tempFile.absolutePath)
                    "json" -> generateReportUseCase.exportToPdf(result, tempFile.absolutePath)
                    else -> Result.failure(Exception("Unknown format"))
                }

                exportResult.fold(
                    onSuccess = { filePath ->
                        // Copy to user-selected location
                        val sourceFile = File(filePath)
                        app.contentResolver.openOutputStream(destinationUri)?.use { outputStream ->
                            sourceFile.inputStream().use { inputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }

                        // Clean up temp file
                        sourceFile.delete()

                        _exportState.value = ExportState.Success("File saved successfully")
                    },
                    onFailure = { error ->
                        _exportState.value = ExportState.Error(error.message ?: "Export failed")
                    }
                )

            } catch (e: Exception) {
                _exportState.value = ExportState.Error(e.message ?: "Save failed")
            }
        }
    }

    /**
     * Share result as text (for messaging apps)
     */
    fun shareResultAsText(result: AnalysisResult) {
        val shareText = buildShareText(result)

        val shareIntent = Intent().apply {
            action = Intent.ACTION_SEND
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "F3Set Tennis Analysis - ${result.videoId}")
            putExtra(Intent.EXTRA_TEXT, shareText)
        }

        _shareState.value = ShareState.ReadyToShare(shareIntent)
    }

    /**
     * Build text content for sharing
     */
    private fun buildShareText(result: AnalysisResult, formatName: String = ""): String {
        val dateFormat = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())

        // Calculate proper confidence from detection confidences
        val detectionConfidences = result.frames.flatMap { it.detections }.map { it.confidence }
        val calculatedConfidence = if (detectionConfidences.isNotEmpty()) {
            detectionConfidences.average().toFloat()
        } else {
            result.confidence
        }

        return buildString {
            appendLine("🎾 F3Set Tennis Analysis Report")
            appendLine("=" .repeat(40))
            appendLine("Video: ${result.videoId}")
            appendLine("Analysis Date: ${dateFormat.format(result.timestamp)}")
            appendLine("Overall Confidence: ${String.format("%.1f%%", calculatedConfidence * 100)}")
            appendLine()

            if (formatName.isNotEmpty()) {
                appendLine("📎 Attached: $formatName")
                appendLine()
            }

            appendLine("📊 Quick Summary:")
            val totalShots = result.frames.count { it.detections.isNotEmpty() }
            appendLine("• ${result.frames.size} frames analyzed")
            appendLine("• $totalShots tennis shots detected")

            if (totalShots > 0) {
                val topActions = result.frames.flatMap { it.detections }
                    .groupBy { it.label }
                    .mapValues { it.value.size }
                    .toList()
                    .sortedByDescending { it.second }
                    .take(3)

                appendLine("• Top actions: ${topActions.joinToString(", ") { "${it.first} (${it.second})" }}")
            }

            appendLine()
            appendLine("🔍 Detailed Summary:")
            appendLine(result.summary.take(300))
            if (result.summary.length > 300) {
                appendLine("...")
            }

            appendLine()
            appendLine("🤖 Generated by AI Video Analyzer (F3Set)")
            appendLine("🔒 All processing done locally on device")
            appendLine()
            appendLine("Note: Confidence calculated from individual detection scores for accuracy")
        }
    }

    fun exportAllResults() {
        viewModelScope.launch {
            _exportState.value = ExportState.Exporting

            try {
                val results = _analysisResults.value ?: emptyList()
                if (results.isEmpty()) {
                    _exportState.value = ExportState.Error("No results to export")
                    return@launch
                }

                val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault())
                val timestamp = dateFormat.format(Date())
                val fileName = "f3set_all_results_${timestamp}.xlsx"
                val tempFile = File(app.cacheDir, fileName)

                generateReportUseCase.exportMultipleResults(
                    results,
                    GenerateReportUseCase.ExportFormat.EXCEL,
                    tempFile.absolutePath
                ).fold(
                    onSuccess = { filePath ->
                        _exportState.value = ExportState.Success(filePath)

                        // Prepare sharing intent for all results
                        val file = File(filePath)
                        if (file.exists()) {
                            prepareShareIntentForAllResults(file)
                        }
                    },
                    onFailure = { error ->
                        _exportState.value = ExportState.Error(error.message ?: "Export failed")
                    }
                )

            } catch (e: Exception) {
                _exportState.value = ExportState.Error(e.message ?: "Export failed")
            }
        }
    }

    /**
     * Prepare sharing intent for all results export
     */
    private fun prepareShareIntentForAllResults(file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                app,
                "${app.packageName}.fileprovider",
                file
            )

            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "F3Set Tennis Analysis - All Results")
                putExtra(Intent.EXTRA_TEXT, "Complete F3Set tennis analysis results exported from AI Video Analyzer")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            _shareState.value = ShareState.ReadyToShare(shareIntent)

        } catch (e: Exception) {
            _shareState.value = ShareState.Error("Failed to prepare sharing: ${e.message}")
        }
    }

    fun clearShareState() {
        _shareState.value = ShareState.Idle
    }
}