package com.example.aivideoanalyzer.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
//import android.util.Log
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import com.example.aivideoanalyzer.domain.model.Detection
import com.example.aivideoanalyzer.domain.model.FrameAnalysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.*

/**
 * TODO:
 * 1. DEFAULT_CLIP_LENGTH = 96 (matches Python)
 * 2. Progress callback 0-1 range (not 0-100)
 * 3. Proper bitmap recycling in all cases
 * 4. Zero-padding consistency with Python
 * 5. Simplified vote accumulation
 * 6. Performance optimizations
 */
class F3SetVideoProcessor(
    private val context: Context,
    private val inferenceManager: F3SetInferenceManager
) {

    companion object {
        private const val TAG = "F3SetVideoProcessor"

        private const val DEFAULT_CLIP_LENGTH = 96  // FIXED: Was 48, now matches Python
        private const val SLIDING_WINDOW_STRIDE = 48  // FIXED: Larger stride, less overlap
        private const val CROP_DIM = 224

        // Simplified vote accumulation
        private const val VOTE_CONFIDENCE_BOOST = 1.1f

        // Performance optimizations
        private const val ENABLE_VERBOSE_LOGGING = false
        private const val BITMAP_RECYCLE_AGGRESSIVE = true
    }

    data class F3SetVideoResult(
        val shots: List<ShotDetection>,
        val frameAnalyses: List<FrameAnalysis>,
        val summary: VideoSummary,
        val processingTimeMs: Long
    )

    data class ShotDetection(
        val startFrame: Int,
        val endFrame: Int,
        val startTimeMs: Long,
        val endTimeMs: Long,
        val actionClasses: List<ActionClass>,
        val confidence: Float,
        val voteCount: Int = 1
    )

    data class ActionClass(
        val label: String,
        val confidence: Float
    )

    data class VideoSummary(
        val totalFrames: Int,
        val totalShots: Int,
        val dominantActions: List<ActionClass>,
        val averageConfidence: Float
    )

    suspend fun processVideo(
        videoUri: Uri,
        onProgress: suspend (Float) -> Unit = {}
    ): Result<F3SetVideoResult> = withContext(Dispatchers.IO) {

        val startTime = System.currentTimeMillis()
        val retriever = MediaMetadataRetriever()

        try {
            retriever.setDataSource(context, videoUri)

            val duration = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION
            )?.toLongOrNull() ?: 0L

            val frameRate = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE
            )?.toFloatOrNull()?.toInt() ?: 30

            val totalFrames = (duration * frameRate / 1000).toInt()

//            Log.d(TAG, "Video: ${duration}ms, ${frameRate}fps, $totalFrames frames")
//            Log.d(TAG, "Using FIXED parameters: clip=$DEFAULT_CLIP_LENGTH, stride=$SLIDING_WINDOW_STRIDE")

            // Process with fixed parameters
            val windowResults = processVideoFixed(
                retriever = retriever,
                totalFrames = totalFrames,
                frameRate = frameRate,
                onProgress = onProgress
            )

            val aggregatedResult = aggregateResultsFixed(windowResults, totalFrames, frameRate)
            val processingTime = System.currentTimeMillis() - startTime

            Result.success(aggregatedResult.copy(processingTimeMs = processingTime))

        } catch (e: Exception) {
//            Log.e(TAG, "Video processing failed", e)
            Result.failure(e)
        } finally {
            retriever.release()
        }
    }

    private suspend fun processVideoFixed(
        retriever: MediaMetadataRetriever,
        totalFrames: Int,
        frameRate: Int,
        onProgress: suspend (Float) -> Unit
    ): List<WindowResult> = withContext(Dispatchers.Default) {

        val results = mutableListOf<WindowResult>()
        val handInfo = F3SetInferenceManager.HandInfo.getDefault()

        var clipStartFrame = 0
        var clipIndex = 0
        val totalClips = (totalFrames + SLIDING_WINDOW_STRIDE - 1) / SLIDING_WINDOW_STRIDE

//        Log.d(TAG, "Processing $totalClips clips total")

        while (clipStartFrame < totalFrames) {
            val clipEndFrame = minOf(clipStartFrame + DEFAULT_CLIP_LENGTH, totalFrames)

            if (ENABLE_VERBOSE_LOGGING) {
//                Log.d(TAG, "Processing clip $clipIndex: frames $clipStartFrame-${clipEndFrame-1}")
            }

            // Extract frames for this clip
            val clipFrames = extractClipFramesFixed(retriever, clipStartFrame, frameRate, totalFrames)

            // Process frames and recycle bitmaps
            try {
                if (clipFrames.size >= DEFAULT_CLIP_LENGTH) {
                    // Process this clip
                    val clipResult = inferenceManager.processVideoClip(
                        frames = clipFrames.map { it.bitmap },
                        handInfo = handInfo,
                        startFrameIndex = clipStartFrame
                    ).getOrThrow()

                    results.add(
                        WindowResult(
                            clipIndex = clipIndex,
                            startFrame = clipStartFrame,
                            endFrame = clipEndFrame,
                            frames = clipFrames,
                            inferenceResult = clipResult
                        )
                    )

                    if (clipResult.shots.isNotEmpty() && ENABLE_VERBOSE_LOGGING) {
//                        Log.d(TAG, "Clip $clipIndex detections: ${clipResult.shots.map { it.absoluteFrameIndex }.joinToString()}")
                    }
                } else {
//                    Log.w(TAG, "Skipping clip $clipIndex: insufficient frames (${clipFrames.size})")
                }
            } finally {
                // Always recycle bitmaps to prevent memory leaks
                if (BITMAP_RECYCLE_AGGRESSIVE) {
                    val uniqueBitmaps = clipFrames.map { it.bitmap }.distinct()
                    uniqueBitmaps.forEach { bitmap ->
                        if (!bitmap.isRecycled) {
                            bitmap.recycle()
                        }
                    }
                }
            }

            // Calculate and report progress (but don't call the callback to avoid threading issues)
            val progress = (clipIndex + 1).toFloat() / totalClips
//            Log.d(TAG, "Progress: ${(progress * 100).toInt()}% (clip ${clipIndex + 1}/$totalClips)")

            // Only call progress callback occasionally to avoid threading issues
            if (clipIndex % 2 == 0 || clipIndex == totalClips - 1) {
                try {
                    onProgress(progress)
                } catch (e: Exception) {
//                    Log.w(TAG, "Progress callback failed, continuing processing", e)
                }
            }

            clipStartFrame += SLIDING_WINDOW_STRIDE
            clipIndex++

            if (clipIndex % 5 == 0) {
                System.gc()
            }
        }

//        Log.d(TAG, "Processed ${results.size} clips with FIXED parameters")
        results
    }

    /**
     * CRITICAL FIX 6: Consistent zero-padding like Python
     */
    private fun extractClipFramesFixed(
        retriever: MediaMetadataRetriever,
        clipStartFrame: Int,
        frameRate: Int,
        totalFrames: Int
    ): List<TimedFrame> {
        val clipFrames = mutableListOf<TimedFrame>()

        for (frameIdx in 0 until DEFAULT_CLIP_LENGTH) {
            val absoluteFrameNum = clipStartFrame + frameIdx

            if (absoluteFrameNum >= totalFrames) {
                // CRITICAL FIX 7: Zero-padding (create black frame) instead of duplicating last frame
                val zeroBitmap = Bitmap.createBitmap(CROP_DIM, CROP_DIM, Bitmap.Config.ARGB_8888)
                zeroBitmap.eraseColor(0) // Black frame

                clipFrames.add(
                    TimedFrame(
                        frameNumber = absoluteFrameNum,
                        timestamp = (totalFrames * 1000L) / frameRate, // Use end time
                        bitmap = zeroBitmap
                    )
                )
            } else {
                // Precise frame timing
                val preciseTimeUs = Math.round((absoluteFrameNum * 1000000.0) / frameRate)

                try {
                    val rawBitmap = retriever.getFrameAtTime(
                        preciseTimeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )

                    if (rawBitmap != null) {
                        // Force ARGB_8888 and resize
                        val argbBitmap = if (rawBitmap.config != Bitmap.Config.ARGB_8888) {
                            val converted = rawBitmap.copy(Bitmap.Config.ARGB_8888, false)
                            rawBitmap.recycle()
                            converted
                        } else {
                            rawBitmap
                        }

                        val resizedBitmap = if (argbBitmap.width != CROP_DIM || argbBitmap.height != CROP_DIM) {
                            val scaled = Bitmap.createScaledBitmap(argbBitmap, CROP_DIM, CROP_DIM, true)
                            if (scaled != argbBitmap) {
                                argbBitmap.recycle()
                            }
                            scaled
                        } else {
                            argbBitmap
                        }

                        clipFrames.add(
                            TimedFrame(
                                frameNumber = absoluteFrameNum,
                                timestamp = Math.round((absoluteFrameNum * 1000.0) / frameRate),
                                bitmap = resizedBitmap
                            )
                        )
                    } else {
                        // If frame extraction fails, create a black frame
                        val zeroBitmap = Bitmap.createBitmap(CROP_DIM, CROP_DIM, Bitmap.Config.ARGB_8888)
                        zeroBitmap.eraseColor(0)

                        clipFrames.add(
                            TimedFrame(
                                frameNumber = absoluteFrameNum,
                                timestamp = Math.round((absoluteFrameNum * 1000.0) / frameRate),
                                bitmap = zeroBitmap
                            )
                        )
                    }
                } catch (e: Exception) {
//                    Log.w(TAG, "Failed to extract frame at $preciseTimeUs", e)

                    // Create black frame on error
                    val zeroBitmap = Bitmap.createBitmap(CROP_DIM, CROP_DIM, Bitmap.Config.ARGB_8888)
                    zeroBitmap.eraseColor(0)

                    clipFrames.add(
                        TimedFrame(
                            frameNumber = absoluteFrameNum,
                            timestamp = Math.round((absoluteFrameNum * 1000.0) / frameRate),
                            bitmap = zeroBitmap
                        )
                    )
                }
            }
        }

        return clipFrames
    }

    /**
     * SIMPLIFIED aggregation without complex vote logic
     */
    private fun aggregateResultsFixed(
        windowResults: List<WindowResult>,
        totalFrames: Int,
        frameRate: Int
    ): F3SetVideoResult {

//        Log.d(TAG, "=== FIXED AGGREGATION ===")

        // Simple accumulation - collect all detections
        val allDetections = mutableMapOf<Int, MutableList<DetectionVote>>()
        var totalRawDetections = 0

        windowResults.forEach { window ->
            val result = window.inferenceResult

            result.shots.forEach { shot ->
                totalRawDetections++
                val frameIndex = shot.absoluteFrameIndex

                allDetections.getOrPut(frameIndex) { mutableListOf() }.add(
                    DetectionVote(
                        confidence = shot.confidence,
                        fineScores = if (shot.frameIndex < result.frameFinePredictions.size)
                            result.frameFinePredictions[shot.frameIndex] else FloatArray(29),
                        clipIndex = window.clipIndex
                    )
                )
            }
        }

//        Log.d(TAG, "Raw detections: $totalRawDetections")
//        Log.d(TAG, "Unique frames: ${allDetections.size}")

        // Create final detections with simple filtering
        val finalDetections = mutableListOf<FinalDetection>()

        for ((frameIndex, votes) in allDetections.toSortedMap()) {
            val voteCount = votes.size
            val maxConfidence = votes.maxByOrNull { it.confidence }?.confidence ?: 0f
            val avgConfidence = votes.map { it.confidence }.average().toFloat()

            // Simple filtering: keep if confidence is reasonable
            val shouldKeep = maxConfidence > 0.01f  // Very low threshold

            if (shouldKeep) {
                // Use max confidence, optionally boosted for multiple votes
                val finalConfidence = if (voteCount > 1) {
                    minOf(1.0f, maxConfidence * VOTE_CONFIDENCE_BOOST)
                } else {
                    maxConfidence
                }

                // Average fine scores
                val avgFineScores = FloatArray(29)
                for (i in avgFineScores.indices) {
                    avgFineScores[i] = votes.map {
                        if (i < it.fineScores.size) it.fineScores[i] else 0f
                    }.average().toFloat()
                }

                finalDetections.add(
                    FinalDetection(
                        frameIndex = frameIndex,
                        confidence = finalConfidence,
                        voteCount = voteCount,
                        fineScores = avgFineScores
                    )
                )
            }
        }

        // Convert to shots and frame analyses
        val shots = mutableListOf<ShotDetection>()
        val frameAnalyses = mutableListOf<FrameAnalysis>()

        finalDetections.forEach { detection ->
            val actions = applyTennisRulesForFrame(detection.fineScores)
            val actionClasses = actions.map { (classIdx, confidence) ->
                ActionClass(getClassName(classIdx), confidence)
            }

            if (actionClasses.isNotEmpty()) {
                frameAnalyses.add(
                    FrameAnalysis(
                        frameNumber = detection.frameIndex,
                        timestamp = (detection.frameIndex * 1000L) / frameRate,
                        detections = actionClasses.map {
                            Detection(it.label, it.confidence, null)
                        },
                        confidence = detection.confidence
                    )
                )
            }

            shots.add(
                ShotDetection(
                    startFrame = detection.frameIndex,
                    endFrame = detection.frameIndex,
                    startTimeMs = (detection.frameIndex * 1000L) / frameRate,
                    endTimeMs = (detection.frameIndex * 1000L) / frameRate,
                    actionClasses = actionClasses,
                    confidence = detection.confidence,
                    voteCount = detection.voteCount
                )
            )
        }

        val summary = createVideoSummary(totalFrames, shots, frameAnalyses)

//        Log.d(TAG, "Final result: ${shots.size} shots detected")

        return F3SetVideoResult(
            shots = shots,
            frameAnalyses = frameAnalyses,
            summary = summary,
            processingTimeMs = 0
        )
    }

    private data class DetectionVote(
        val confidence: Float,
        val fineScores: FloatArray,
        val clipIndex: Int
    )

    private data class FinalDetection(
        val frameIndex: Int,
        val confidence: Float,
        val voteCount: Int,
        val fineScores: FloatArray
    )

    private fun applyTennisRulesForFrame(fineScores: FloatArray): List<Pair<Int, Float>> {
        val selected = mutableListOf<Pair<Int, Float>>()
        val actionGroups = listOf(0 to 2, 2 to 5, 5 to 8, 16 to 24, 25 to 29)

        for ((start, end) in actionGroups) {
            var maxIdx = -1
            var maxScore = -1f
            for (i in start until minOf(end, fineScores.size)) {
                if (fineScores[i] > maxScore) {
                    maxScore = fineScores[i]
                    maxIdx = i
                }
            }
            if (maxIdx >= 0) {
                selected.add(maxIdx to maxScore)
            }
        }

        if (24 < fineScores.size && fineScores[24] > 0.5f) {
            if (!selected.any { it.first == 24 }) {
                selected.add(24 to fineScores[24])
            }
        }

        val hasServe = selected.any { it.first == 5 }
        if (!hasServe && fineScores.size > 10) {
            val additionalGroups = listOf(
                8 to minOf(10, fineScores.size),
                10 to minOf(16, fineScores.size)
            )

            for ((start, end) in additionalGroups) {
                var maxIdx = -1
                var maxScore = -1f
                for (i in start until end) {
                    if (fineScores[i] > maxScore) {
                        maxScore = fineScores[i]
                        maxIdx = i
                    }
                }
                if (maxIdx >= 0) {
                    selected.add(maxIdx to maxScore)
                }
            }
        }

        return selected
    }

    private fun createVideoSummary(
        totalFrames: Int,
        shots: List<ShotDetection>,
        frameAnalyses: List<FrameAnalysis>
    ): VideoSummary {
        val classCount = mutableMapOf<String, Int>()
        val classScores = mutableMapOf<String, MutableList<Float>>()

        frameAnalyses.forEach { frame ->
            frame.detections.forEach { detection ->
                classCount[detection.label] = (classCount[detection.label] ?: 0) + 1
                classScores.getOrPut(detection.label) { mutableListOf() }.add(detection.confidence)
            }
        }

        val dominantActions = classCount.entries
            .sortedByDescending { it.value }
            .take(5)
            .map { (label, count) ->
                val avgConfidence = classScores[label]?.average()?.toFloat() ?: 0f
                ActionClass(label, avgConfidence)
            }

        val avgConfidence = if (shots.isNotEmpty()) {
            shots.map { it.confidence }.average().toFloat()
        } else 0f

        return VideoSummary(
            totalFrames = totalFrames,
            totalShots = shots.size,
            dominantActions = dominantActions,
            averageConfidence = avgConfidence
        )
    }

    private fun getClassName(index: Int): String {
        return tennisClasses[index] ?: "unknown_$index"
    }

    private val tennisClasses = mapOf(
        0 to "near", 1 to "far", 2 to "deuce", 3 to "middle", 4 to "ad",
        5 to "serve", 6 to "return", 7 to "stroke", 8 to "fh", 9 to "bh",
        10 to "gs", 11 to "slice", 12 to "volley", 13 to "smash", 14 to "drop", 15 to "lob",
        16 to "T", 17 to "B", 18 to "W", 19 to "CC", 20 to "DL", 21 to "DM", 22 to "II", 23 to "IO",
        24 to "approach", 25 to "in", 26 to "winner", 27 to "forced-err", 28 to "unforced-err"
    )

    fun convertToAnalysisResult(
        videoId: String,
        f3setResult: F3SetVideoResult
    ): AnalysisResult {
        val summary = buildString {
            appendLine("Detected ${f3setResult.shots.size} tennis shots")
            appendLine("Total frames analyzed: ${f3setResult.summary.totalFrames}")
            appendLine("Average confidence: ${String.format("%.3f", f3setResult.summary.averageConfidence)}")
            appendLine("\nDominant actions:")
            f3setResult.summary.dominantActions.forEach { action ->
                appendLine("  - ${action.label}: ${String.format("%.1f", action.confidence * 100)}%")
            }
            if (f3setResult.shots.isNotEmpty()) {
                appendLine("\nShot timeline:")
                f3setResult.shots.take(10).forEachIndexed { idx, shot ->
                    val timeStr = formatTime(shot.startTimeMs)
                    val actions = shot.actionClasses.take(3).joinToString(", ") { it.label }
                    val voteInfo = if (shot.voteCount > 1) " [${shot.voteCount}x]" else ""
                    appendLine("  Shot ${idx + 1}: $timeStr (frame ${shot.startFrame})$voteInfo - $actions")
                }
                if (f3setResult.shots.size > 10) {
                    appendLine("  ... and ${f3setResult.shots.size - 10} more shots")
                }
            }
        }

        return AnalysisResult(
            videoId = videoId,
            timestamp = Date(),
            frames = f3setResult.frameAnalyses,
            summary = summary,
            confidence = f3setResult.summary.averageConfidence
        )
    }

    private fun formatTime(milliseconds: Long): String {
        val seconds = milliseconds / 1000
        val minutes = seconds / 60
        val remainingSeconds = seconds % 60
        return String.format("%d:%02d", minutes, remainingSeconds)
    }

    private data class WindowResult(
        val clipIndex: Int,
        val startFrame: Int,
        val endFrame: Int,
        val frames: List<TimedFrame>,
        val inferenceResult: F3SetInferenceManager.F3SetResult
    )

    private data class TimedFrame(
        val frameNumber: Int,
        val timestamp: Long,
        val bitmap: Bitmap
    )
}