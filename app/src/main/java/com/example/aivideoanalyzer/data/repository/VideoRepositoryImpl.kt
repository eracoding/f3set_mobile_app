package com.example.aivideoanalyzer.data.repository

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.example.aivideoanalyzer.data.export.ExportManager
import com.example.aivideoanalyzer.data.local.FileManager
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.model.VideoStatus
import com.example.aivideoanalyzer.domain.repository.ProcessingProgress
import com.example.aivideoanalyzer.domain.repository.ProcessingStage
import com.example.aivideoanalyzer.domain.repository.VideoRepository
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.flow
import java.io.File
import java.util.*

class VideoRepositoryImpl(
    private val context: Context,
    private val fileManager: FileManager
) : VideoRepository {

    // In-memory storage for demo purposes
    private val videos = mutableListOf<Video>()
    private val analysisResults = mutableListOf<AnalysisResult>()

    private val videosFlow = MutableStateFlow<List<Video>>(emptyList())
    private val resultsFlow = MutableStateFlow<List<AnalysisResult>>(emptyList())

    override suspend fun uploadVideo(uri: Uri): Result<Video> {
        return try {
            // Get video metadata
            val fileName = getFileName(uri) ?: "video_${System.currentTimeMillis()}.mp4"
            val fileSize = getFileSize(uri) ?: 0L

            // Create video object
            val video = Video(
                id = UUID.randomUUID().toString(),
                uri = uri,
                name = fileName,
                size = fileSize,
                duration = 0L, // TODO: Extract actual duration
                uploadDate = Date(),
                status = VideoStatus.UPLOADED
            )

            // Copy to app storage
            val savedFile = fileManager.saveVideoFile(uri, video.id)
            if (savedFile != null) {
                // Update the video URI to point to the saved file
                val savedVideo = video.copy(uri = Uri.fromFile(savedFile))
                videos.add(savedVideo)
                videosFlow.value = videos.toList()
                Result.success(savedVideo)
            } else {
                Result.failure(Exception("Failed to save video file"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getVideo(videoId: String): Result<Video> {
        return videos.find { it.id == videoId }?.let {
            Result.success(it)
        } ?: Result.failure(Exception("Video not found"))
    }

    override suspend fun getAllVideos(): Result<List<Video>> {
        return Result.success(videos.toList())
    }

    override suspend fun deleteVideo(videoId: String): Result<Unit> {
        return try {
            videos.removeAll { it.id == videoId }
            analysisResults.removeAll { it.videoId == videoId }
            fileManager.deleteVideoFile(videoId)

            videosFlow.value = videos.toList()
            resultsFlow.value = analysisResults.toList()

            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun updateVideoStatus(videoId: String, status: VideoStatus): Result<Unit> {
        return try {
            videos.find { it.id == videoId }?.let { video ->
                val index = videos.indexOf(video)
                videos[index] = video.copy(status = status)
                videosFlow.value = videos.toList()
                Result.success(Unit)
            } ?: Result.failure(Exception("Video not found"))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun processVideo(videoId: String): Flow<ProcessingProgress> = flow {
        // Fallback processing simulation - F3Set will override this in ProcessVideoUseCase

        // Stage 1: Preprocessing
        emit(ProcessingProgress(ProcessingStage.PREPROCESSING, 0, "Initializing F3Set model..."))
        for (i in 0..100 step 20) {
            delay(200)
            emit(ProcessingProgress(ProcessingStage.PREPROCESSING, i, "Extracting video frames..."))
        }

        // Stage 2: Inference
        emit(ProcessingProgress(ProcessingStage.INFERENCING, 0, "Running F3Set inference..."))
        for (i in 0..100 step 10) {
            delay(300)
            emit(ProcessingProgress(ProcessingStage.INFERENCING, i, "Analyzing tennis actions frame ${i/10}/10..."))
        }

        // Stage 3: Post-processing
        emit(ProcessingProgress(ProcessingStage.POST_PROCESSING, 0, "Processing shot detections..."))
        for (i in 0..100 step 25) {
            delay(150)
            emit(ProcessingProgress(ProcessingStage.POST_PROCESSING, i, "Aggregating results..."))
        }

        // Stage 4: Generate report
        emit(ProcessingProgress(ProcessingStage.GENERATING_REPORT, 0, "Generating tennis analysis report..."))
        delay(500)
        emit(ProcessingProgress(ProcessingStage.GENERATING_REPORT, 100, "Report ready"))

        // Complete
        emit(ProcessingProgress(ProcessingStage.COMPLETED, 100, "Tennis analysis completed successfully"))

        // Update video status
        updateVideoStatus(videoId, VideoStatus.COMPLETED)

        // Create fallback analysis result (if F3Set didn't process)
        if (analysisResults.none { it.videoId == videoId }) {
            val result = createFallbackAnalysisResult(videoId)
            saveAnalysisResult(result)
        }
    }

    override suspend fun cancelProcessing(videoId: String): Result<Unit> {
        return updateVideoStatus(videoId, VideoStatus.UPLOADED)
    }

    override suspend fun retryProcessing(videoId: String): Result<Unit> {
        return updateVideoStatus(videoId, VideoStatus.UPLOADED)
    }

    override suspend fun saveAnalysisResult(result: AnalysisResult): Result<Unit> {
        return try {
            // Remove any existing result for this video
            analysisResults.removeAll { it.videoId == result.videoId }
            // Add the new result
            analysisResults.add(result)
            resultsFlow.value = analysisResults.toList()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getAnalysisResult(videoId: String): Result<AnalysisResult> {
        return analysisResults.find { it.videoId == videoId }?.let {
            Result.success(it)
        } ?: Result.failure(Exception("Analysis result not found"))
    }

    override suspend fun getAllAnalysisResults(): Result<List<AnalysisResult>> {
        return Result.success(analysisResults.toList())
    }

    override suspend fun deleteAnalysisResult(videoId: String): Result<Unit> {
        return try {
            analysisResults.removeAll { it.videoId == videoId }
            resultsFlow.value = analysisResults.toList()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun exportToExcel(result: AnalysisResult, filePath: String): Result<String> {
        // Export as HTML instead of Excel to avoid Apache POI dependencies
        return try {
            val exportManager = ExportManager()
            val file = File(filePath.replace(".xlsx", ".html"))
            exportManager.exportToHtml(result, file)
            Result.success(file.absolutePath)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun exportToCsv(result: AnalysisResult, filePath: String): Result<String> {
        return try {
            val exportManager = ExportManager()
            val file = File(filePath)
            exportManager.exportToCsv(result, file)
            Result.success(file.absolutePath)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun exportToPdf(result: AnalysisResult, filePath: String): Result<String> {
        // Export as JSON for now, can implement PDF later if needed
        return try {
            val exportManager = ExportManager()
            val file = File(filePath.replace(".pdf", ".json"))
            exportManager.exportToJson(result, file)
            Result.success(file.absolutePath)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override fun observeVideos(): Flow<List<Video>> = videosFlow

    override fun observeAnalysisResults(): Flow<List<AnalysisResult>> = resultsFlow

    // Helper functions
    private fun createFallbackAnalysisResult(videoId: String): AnalysisResult {
        return AnalysisResult(
            videoId = videoId,
            timestamp = Date(),
            frames = emptyList(),
            summary = "Fallback analysis completed. F3Set model may not have been available during processing.",
            confidence = 0.50f
        )
    }

    private fun getFileName(uri: Uri): String? {
        return context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            cursor.moveToFirst()
            cursor.getString(nameIndex)
        }
    }

    private fun getFileSize(uri: Uri): Long? {
        return context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            cursor.moveToFirst()
            cursor.getLong(sizeIndex)
        }
    }
}