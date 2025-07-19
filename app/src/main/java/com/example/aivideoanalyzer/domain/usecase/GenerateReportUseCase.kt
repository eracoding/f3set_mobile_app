package com.example.aivideoanalyzer.domain.usecase

import com.example.aivideoanalyzer.data.export.ExportManager
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import java.io.File

class GenerateReportUseCase(
) {
    private val exportManager = ExportManager()

    suspend fun exportToExcel(
        result: AnalysisResult,
        outputPath: String
    ): Result<String> {
        return try {
            val outputFile = File(outputPath)
            // Note: This actually exports to HTML format (rich report)
            exportManager.exportToHtml(result, outputFile).fold(
                onSuccess = { file ->
                    Result.success(file.absolutePath)
                },
                onFailure = { error ->
                    Result.failure(error)
                }
            )
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun exportToCsv(
        result: AnalysisResult,
        outputPath: String
    ): Result<String> {
        return try {
            val outputFile = File(outputPath)
            exportManager.exportToCsv(result, outputFile).fold(
                onSuccess = { file ->
                    Result.success(file.absolutePath)
                },
                onFailure = { error ->
                    Result.failure(error)
                }
            )
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun exportToPdf(
        result: AnalysisResult,
        outputPath: String
    ): Result<String> {
        return try {
            val outputFile = File(outputPath)
            // Note: This actually exports to JSON format (structured data)
            exportManager.exportToJson(result, outputFile).fold(
                onSuccess = { file ->
                    Result.success(file.absolutePath)
                },
                onFailure = { error ->
                    Result.failure(error)
                }
            )
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun exportMultipleResults(
        results: List<AnalysisResult>,
        format: ExportFormat,
        outputPath: String
    ): Result<String> {
        return try {
            val outputFile = File(outputPath)

            when (format) {
                ExportFormat.EXCEL -> {
                    // Export consolidated HTML report for multiple results
                    exportManager.exportMultipleResults(results, outputFile).fold(
                        onSuccess = { file ->
                            Result.success(file.absolutePath)
                        },
                        onFailure = { error ->
                            Result.failure(error)
                        }
                    )
                }
                ExportFormat.CSV -> {
                    // Export consolidated CSV for multiple results
                    exportMultipleResultsToCsv(results, outputFile)
                }
                ExportFormat.PDF -> {
                    // Export consolidated JSON for multiple results
                    exportMultipleResultsToJson(results, outputFile)
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun exportMultipleResultsToCsv(results: List<AnalysisResult>, outputFile: File): Result<String> {
        return try {
            val consolidatedContent = buildString {
                appendLine("# F3Set Tennis Analysis - Multiple Results Export")
                appendLine("# Generated: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())}")
                appendLine("# Total Videos: ${results.size}")
                appendLine("")

                // Headers with corrected column names
                appendLine("Video ID,Analysis Date,Calculated Confidence,Total Frames,Tennis Shots,Individual Detections,Frame Number,Timestamp (ms),Action Type,Detection Confidence,Bounding Box X,Bounding Box Y,Bounding Box Width,Bounding Box Height")

                // Data rows with corrected statistics
                results.forEach { result ->
                    val dateStr = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(result.timestamp)
                    val totalFrames = result.frames.size
                    val totalShots = result.frames.count { it.detections.isNotEmpty() }
                    val totalDetections = result.frames.sumOf { it.detections.size }

                    // Calculate proper confidence
                    val detectionConfidences = result.frames.flatMap { it.detections }.map { it.confidence }
                    val calculatedConfidence = if (detectionConfidences.isNotEmpty()) {
                        detectionConfidences.average().toFloat()
                    } else {
                        result.confidence
                    }

                    result.frames.forEach { frame ->
                        if (frame.detections.isNotEmpty()) {
                            frame.detections.forEach { detection ->
                                val bbox = detection.boundingBox
                                val line = listOf(
                                    "\"${result.videoId}\"",
                                    "\"$dateStr\"",
                                    String.format("%.4f", calculatedConfidence),
                                    totalFrames.toString(),
                                    totalShots.toString(),
                                    totalDetections.toString(),
                                    frame.frameNumber.toString(),
                                    frame.timestamp.toString(),
                                    "\"${detection.label}\"",
                                    String.format("%.4f", detection.confidence),
                                    bbox?.x?.toString() ?: "",
                                    bbox?.y?.toString() ?: "",
                                    bbox?.width?.toString() ?: "",
                                    bbox?.height?.toString() ?: ""
                                ).joinToString(",")
                                appendLine(line)
                            }
                        } else {
                            // Include videos with no detections
                            val line = listOf(
                                "\"${result.videoId}\"",
                                "\"$dateStr\"",
                                String.format("%.4f", calculatedConfidence),
                                totalFrames.toString(),
                                totalShots.toString(),
                                totalDetections.toString(),
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                ""
                            ).joinToString(",")
                            appendLine(line)
                        }
                    }
                }
            }

            outputFile.writeText(consolidatedContent)
            Result.success(outputFile.absolutePath)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun exportMultipleResultsToJson(results: List<AnalysisResult>, outputFile: File): Result<String> {
        return try {
            val dateFormat = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault())

            val json = buildString {
                appendLine("{")
                appendLine("  \"metadata\": {")
                appendLine("    \"generator\": \"AI Video Analyzer - F3Set Tennis Action Recognition\",")
                appendLine("    \"version\": \"1.0.0\",")
                appendLine("    \"exportDate\": \"${dateFormat.format(java.util.Date())}\",")
                appendLine("    \"exportType\": \"multiple_results\",")
                appendLine("    \"totalVideos\": ${results.size}")
                appendLine("  },")
                appendLine("  \"consolidatedStatistics\": {")
                appendLine("    \"totalFrames\": ${results.sumOf { it.frames.size }},")
                appendLine("    \"totalDetections\": ${results.sumOf { it.frames.sumOf { frame -> frame.detections.size } }},")
                appendLine("    \"averageConfidence\": ${results.map { it.confidence }.average()},")
                appendLine("    \"videoCount\": ${results.size}")
                appendLine("  },")
                appendLine("  \"results\": [")

                results.forEachIndexed { index, result ->
                    appendLine("    {")
                    appendLine("      \"videoId\": \"${result.videoId}\",")
                    appendLine("      \"timestamp\": \"${dateFormat.format(result.timestamp)}\",")
                    appendLine("      \"confidence\": ${result.confidence},")
                    appendLine("      \"summary\": \"${result.summary.replace("\"", "\\\"").replace("\n", "\\n")}\",")
                    appendLine("      \"frameCount\": ${result.frames.size},")
                    appendLine("      \"detectionCount\": ${result.frames.sumOf { it.detections.size }},")
                    appendLine("      \"frames\": [")

                    result.frames.forEachIndexed { frameIndex, frame ->
                        if (frame.detections.isNotEmpty()) {
                            appendLine("        {")
                            appendLine("          \"frameNumber\": ${frame.frameNumber},")
                            appendLine("          \"timestamp\": ${frame.timestamp},")
                            appendLine("          \"confidence\": ${frame.confidence},")
                            appendLine("          \"detections\": [")

                            frame.detections.forEachIndexed { detIndex, detection ->
                                appendLine("            {")
                                appendLine("              \"action\": \"${detection.label}\",")
                                appendLine("              \"confidence\": ${detection.confidence}")

                                detection.boundingBox?.let { bbox ->
                                    appendLine("              ,\"boundingBox\": {")
                                    appendLine("                \"x\": ${bbox.x},")
                                    appendLine("                \"y\": ${bbox.y},")
                                    appendLine("                \"width\": ${bbox.width},")
                                    appendLine("                \"height\": ${bbox.height}")
                                    appendLine("              }")
                                }

                                append("            }")
                                if (detIndex < frame.detections.size - 1) append(",")
                                appendLine()
                            }

                            appendLine("          ]")
                            append("        }")
                            if (frameIndex < result.frames.size - 1 && result.frames.drop(frameIndex + 1).any { it.detections.isNotEmpty() }) {
                                append(",")
                            }
                            appendLine()
                        }
                    }

                    appendLine("      ]")
                    append("    }")
                    if (index < results.size - 1) append(",")
                    appendLine()
                }

                appendLine("  ]")
                appendLine("}")
            }

            outputFile.writeText(json)
            Result.success(outputFile.absolutePath)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    enum class ExportFormat {
        EXCEL, CSV, PDF
    }
}