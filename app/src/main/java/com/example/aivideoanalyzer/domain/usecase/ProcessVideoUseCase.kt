package com.example.aivideoanalyzer.domain.usecase

import com.example.aivideoanalyzer.domain.repository.ProcessingProgress
import com.example.aivideoanalyzer.domain.repository.ProcessingStage
import com.example.aivideoanalyzer.domain.repository.VideoRepository
import com.example.aivideoanalyzer.ml.F3SetInferenceManager
import com.example.aivideoanalyzer.ml.F3SetVideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext

class ProcessVideoUseCase(
    private val repository: VideoRepository,
    private val f3setInferenceManager: F3SetInferenceManager? = null,
    private val f3setVideoProcessor: F3SetVideoProcessor? = null
) {
    suspend operator fun invoke(videoId: String): Flow<ProcessingProgress> {
        // If F3Set model is available, use it
        if (f3setInferenceManager != null && f3setVideoProcessor != null) {
            return processWithF3Set(videoId)
        }

        // Otherwise use default processing
        return repository.processVideo(videoId)
    }

    private fun processWithF3Set(videoId: String): Flow<ProcessingProgress> = flow {
        try {
            // Get video details
            val video = repository.getVideo(videoId).getOrThrow()

            // Update status
            repository.updateVideoStatus(videoId, com.example.aivideoanalyzer.domain.model.VideoStatus.PREPROCESSING)

            emit(ProcessingProgress(
                stage = ProcessingStage.PREPROCESSING,
                progress = 0,
                message = "Starting F3Set model processing..."
            ))

            // Load model
            emit(ProcessingProgress(
                stage = ProcessingStage.PREPROCESSING,
                progress = 20,
                message = "Loading F3Set AI model..."
            ))

            f3setInferenceManager?.loadModel()?.getOrThrow()
                ?: throw IllegalStateException("F3Set manager is null")

            emit(ProcessingProgress(
                stage = ProcessingStage.PREPROCESSING,
                progress = 50,
                message = "F3Set model loaded successfully"
            ))

            emit(ProcessingProgress(
                stage = ProcessingStage.PREPROCESSING,
                progress = 80,
                message = "Extracting video frames..."
            ))

            // Update status to processing
            repository.updateVideoStatus(videoId, com.example.aivideoanalyzer.domain.model.VideoStatus.PROCESSING)

            emit(ProcessingProgress(
                stage = ProcessingStage.INFERENCING,
                progress = 10,
                message = "Initializing F3Set video processing..."
            ))

            // Process video WITHOUT progress callback to avoid threading issues
            val result = f3setVideoProcessor?.processVideo(video.uri)?.getOrThrow()
                ?: throw IllegalStateException("F3Set processor is null")

            // Emit progress milestones during processing
            emit(ProcessingProgress(
                stage = ProcessingStage.INFERENCING,
                progress = 30,
                message = "Processing video clips..."
            ))

            emit(ProcessingProgress(
                stage = ProcessingStage.INFERENCING,
                progress = 60,
                message = "Analyzing tennis actions..."
            ))

            emit(ProcessingProgress(
                stage = ProcessingStage.INFERENCING,
                progress = 90,
                message = "Detecting tennis shots..."
            ))

            // Post-processing
            emit(ProcessingProgress(
                stage = ProcessingStage.POST_PROCESSING,
                progress = 20,
                message = "Converting F3Set results..."
            ))

            val analysisResult = f3setVideoProcessor.convertToAnalysisResult(videoId, result)

            emit(ProcessingProgress(
                stage = ProcessingStage.POST_PROCESSING,
                progress = 70,
                message = "Saving analysis results..."
            ))

            repository.saveAnalysisResult(analysisResult)

            emit(ProcessingProgress(
                stage = ProcessingStage.GENERATING_REPORT,
                progress = 90,
                message = "Finalizing tennis shot analysis..."
            ))

            // Update status
            repository.updateVideoStatus(videoId, com.example.aivideoanalyzer.domain.model.VideoStatus.COMPLETED)

            emit(ProcessingProgress(
                stage = ProcessingStage.COMPLETED,
                progress = 100,
                message = "Analysis completed! Found ${result.shots.size} tennis shots."
            ))

        } catch (e: Exception) {
            // Update status and re-throw
            withContext(Dispatchers.IO) {
                repository.updateVideoStatus(videoId, com.example.aivideoanalyzer.domain.model.VideoStatus.ERROR)
            }
            throw e
        }
    }.flowOn(Dispatchers.IO)


    suspend fun cancel(videoId: String): Result<Unit> {
        return repository.cancelProcessing(videoId)
    }

    suspend fun retry(videoId: String): Result<Unit> {
        return repository.retryProcessing(videoId)
    }
}