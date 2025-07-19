package com.example.aivideoanalyzer.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.aivideoanalyzer.domain.model.Detection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream


class F3SetInferenceManager(private val context: Context) {

    companion object {
        private const val TAG = "F3SetInference"

        // Model files
        private const val MODEL_FILE = "pad/model_scripted.pt"

        // Model parameters
        private const val INPUT_SIZE = 224
        private const val CLIP_LEN = 48
        private const val NUM_CLASSES = 29

        // ImageNet normalization
        private val NORM_MEAN_RGB = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val NORM_STD_RGB = floatArrayOf(0.229f, 0.224f, 0.225f)

        // RELAXED DETECTION THRESHOLDS
        private const val STRONG_THRESHOLD = 0.2f     // Lowered from 0.3f
        private const val MEDIUM_THRESHOLD = 0.03f    // Lowered from 0.08f
        private const val LOCAL_MAX_MIN = 0.01f       // Lowered from 0.02f
        private const val PEAK_STRENGTH_MIN = 0.003f  // Lowered from 0.005f
        private const val RELATIVE_THRESHOLD = 0.08f  // Lowered from 0.12f
        private const val CONTEXTUAL_MIN = 0.02f      // Lowered from 0.03f
        private const val CONTEXTUAL_MULTIPLIER = 2.0f // Lowered from 2.5f

        // Enable/disable criteria for tuning
        private const val ENABLE_STRONG = true
        private const val ENABLE_MEDIUM = true
        private const val ENABLE_LOCAL_MAX = true
        private const val ENABLE_RELATIVE = true      // Keep enabled but lowered
        private const val ENABLE_CONTEXTUAL = true
        private const val ENABLE_EARLY_EXIT = true    // New: fg > bg shortcut
    }

    private var module: Module? = null
    private var isModelLoaded = false

    data class F3SetResult(
        val frameCoarsePredictions: IntArray,
        val frameCoarseScores: Array<FloatArray>,
        val frameFinePredictions: Array<FloatArray>,
        val shots: List<ShotDetection>,
        val inferenceTimeMs: Long,
        val clipStartFrame: Int = 0,
        val debugInfo: DebugInfo = DebugInfo()
    )

    data class ShotDetection(
        val frameIndex: Int,
        val absoluteFrameIndex: Int,
        val actionClasses: List<Detection>,
        val confidence: Float,
        val detectionReason: String = ""
    )

    data class DebugInfo(
        val fgCurve: List<Pair<Int, Float>> = emptyList(),
        val criteriaResults: List<CriteriaDebug> = emptyList()
    )

    data class CriteriaDebug(
        val frameIndex: Int,
        val fgScore: Float,
        val bgScore: Float,
        val strong: Boolean,
        val medium: Boolean,
        val localMax: Boolean,
        val relative: Boolean,
        val contextual: Boolean,
        val earlyExit: Boolean,
        val finalDecision: Boolean
    )

    data class HandInfo(
        val farHand: HandType,
        val nearHand: HandType
    ) {
        enum class HandType { LEFT, RIGHT }

        fun toTensor(): FloatArray {
            return floatArrayOf(
                if (farHand == HandType.LEFT) 1.0f else 0.0f,
                if (nearHand == HandType.LEFT) 1.0f else 0.0f
            )
        }

        companion object {
            fun getDefault() = HandInfo(
                farHand = HandType.RIGHT,
                nearHand = HandType.RIGHT
            )
        }
    }

    suspend fun loadModel(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
////            Log.d(TAG, "Loading F3Set model...")
            val modelPath = copyAssetToFile(MODEL_FILE)
            module = Module.load(modelPath)
            isModelLoaded = true
////            Log.d(TAG, "Model loaded successfully")
            Result.success(Unit)
        } catch (e: Exception) {
////            Log.e(TAG, "Failed to load model", e)
            Result.failure(e)
        }
    }

    suspend fun processVideoClip(
        frames: List<Bitmap>,
        handInfo: HandInfo? = null,
        startFrameIndex: Int = 0
    ): Result<F3SetResult> = withContext(Dispatchers.Default) {

        if (!isModelLoaded) {
            return@withContext Result.failure(IllegalStateException("Model not loaded"))
        }

        try {
            val startTime = System.currentTimeMillis()

            // Prepare inputs
            val frameTensor = prepareBatchFramesSimplified(frames)
            val handTensor = prepareHandTensor(handInfo ?: HandInfo.getDefault())

            // Run inference
            val outputs = module!!.forward(
                IValue.from(frameTensor),
                IValue.from(handTensor)
            )

            if (!outputs.isTuple || outputs.toTuple().size != 3) {
                return@withContext Result.failure(
                    IllegalStateException("Expected 3 outputs from model")
                )
            }

            val outputTuple = outputs.toTuple()
            val coarseProb = outputTuple[1].toTensor()
            val fineProb = outputTuple[2].toTensor()

            // Debug model outputs with FG curve logging
//            debugModelOutputsDetailed(coarseProb, startFrameIndex)

            // Parse with relaxed detection logic
            val result = parseModelOutputRelaxed(
                coarseProb, fineProb,
                frames.size, startFrameIndex
            )

            val inferenceTime = System.currentTimeMillis() - startTime

////            Log.d(TAG, "Inference complete: ${result.shots.size} shots detected in ${inferenceTime}ms")

            Result.success(result.copy(inferenceTimeMs = inferenceTime))

        } catch (e: Exception) {
//            Log.e(TAG, "Inference failed", e)
            e.printStackTrace()
            Result.failure(e)
        }
    }

    private fun prepareBatchFramesSimplified(frames: List<Bitmap>): Tensor {
        val actualFrames = frames.size
        val numFrames = CLIP_LEN

        // Tensor shape: [1, 48, 3, 224, 224]
        val tensorData = FloatArray(1 * numFrames * 3 * INPUT_SIZE * INPUT_SIZE)

        for (i in 0 until numFrames) {
            if (i < actualFrames) {
                val bitmap = frames[i]

                // Convert to tensor with ImageNet normalization
                val frameTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap, NORM_MEAN_RGB, NORM_STD_RGB
                )

                // Copy frame data to batch tensor
                val frameData = frameTensor.dataAsFloatArray
                System.arraycopy(
                    frameData, 0,
                    tensorData, i * 3 * INPUT_SIZE * INPUT_SIZE,
                    frameData.size
                )
            }
        }

        return Tensor.fromBlob(
            tensorData,
            longArrayOf(1, numFrames.toLong(), 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
    }

    private fun debugModelOutputsDetailed(coarseProb: Tensor, startFrameIndex: Int) {
//        Log.d(TAG, "=== DETAILED MODEL OUTPUT ANALYSIS ===")

        val probData = extractFloatData(coarseProb)
        var potentialCount = 0
        var strongCount = 0

//        // Log for plotting (FG_CURVE tag for easy filtering)
        for (i in 0 until minOf(probData.size / 2, CLIP_LEN)) {
            val bgProb = probData[i * 2]
            val fgProb = probData[i * 2 + 1]
            val frameNum = i + startFrameIndex

            // CSV format for easy plotting: frame,fg_prob,bg_prob,fg_ratio
            val fgRatio = fgProb / (fgProb + bgProb)
//            Log.d("FG_CURVE", "$frameNum,$fgProb,$bgProb,$fgRatio")

            if (fgProb > bgProb) {
                strongCount++
//                Log.d(TAG, "Frame $frameNum: 🎯 STRONG! fg=${String.format("%.6f", fgProb)} > bg=${String.format("%.6f", bgProb)}")
            } else if (fgProb > MEDIUM_THRESHOLD) {
                potentialCount++
//                Log.d(TAG, "Frame $frameNum: 🔍 POTENTIAL! fg=${String.format("%.6f", fgProb)} (${String.format("%.1f", fgRatio * 100)}%)")
            } else if (fgProb > 0.01f) {
//                Log.d(TAG, "Frame $frameNum: ⚪ weak fg=${String.format("%.6f", fgProb)} (${String.format("%.1f", fgRatio * 100)}%)")
            }
        }

//        Log.d(TAG, "Summary: $strongCount strong (fg>bg), $potentialCount potential (fg>${MEDIUM_THRESHOLD})")
    }

    private fun prepareHandTensor(handInfo: HandInfo): Tensor {
        val handData = handInfo.toTensor()
        return Tensor.fromBlob(handData, longArrayOf(1, 2))
    }

    /**
     * RELAXED parsing with lower thresholds and better debugging
     */
    private fun parseModelOutputRelaxed(
        coarseProb: Tensor,
        fineProb: Tensor,
        numFrames: Int,
        startFrameIndex: Int
    ): F3SetResult {

//        Log.d(TAG, "=== RELAXED PARSING WITH LOWER THRESHOLDS ===")

        val coarseProbData = extractFloatData(coarseProb)
        val fineProbData = extractFloatData(fineProb)
        val actualFrames = minOf(numFrames, CLIP_LEN)

        // Initialize result arrays
        val coarsePredictions = IntArray(actualFrames)
        val coarseScores = Array(actualFrames) { FloatArray(2) }
        val finePredictions = Array(actualFrames) { FloatArray(NUM_CLASSES) }

        // Debug info
        val fgCurve = mutableListOf<Pair<Int, Float>>()
        val criteriaResults = mutableListOf<CriteriaDebug>()

        // Fill arrays with raw scores
        for (i in 0 until actualFrames) {
            val coarseOffset = i * 2
            if (coarseOffset + 1 < coarseProbData.size) {
                coarseScores[i][0] = coarseProbData[coarseOffset]     // background
                coarseScores[i][1] = coarseProbData[coarseOffset + 1] // foreground

                fgCurve.add(Pair(startFrameIndex + i, coarseScores[i][1]))
            }

            val fineOffset = i * NUM_CLASSES
            if (fineOffset + NUM_CLASSES <= fineProbData.size) {
                for (j in 0 until NUM_CLASSES) {
                    finePredictions[i][j] = fineProbData[fineOffset + j]
                }
            }
        }

        // Apply relaxed detection logic with detailed debugging
        val (detectionResults, debugResults) = detectShotsRelaxed(coarseScores, startFrameIndex)
        for (i in detectionResults.indices) {
            coarsePredictions[i] = detectionResults[i]
        }
        criteriaResults.addAll(debugResults)

        val parsedDetections = coarsePredictions.count { it == 1 }
//        Log.d(TAG, "🎯 RELAXED DETECTIONS: $parsedDetections")

        if (parsedDetections > 0) {
            val detectionIndices = coarsePredictions.toList().mapIndexedNotNull { index, value ->
                if (value == 1) index + startFrameIndex else null
            }
//            Log.d(TAG, "🎯 Relaxed detection frames: ${detectionIndices.joinToString()}")
        }

        // Build shots from detections
        val shots = mutableListOf<ShotDetection>()

        for (i in 0 until actualFrames) {
            if (coarsePredictions[i] == 1) {
                val absoluteFrame = startFrameIndex + i

                val actions = applyTennisRules(finePredictions[i])

                // Get detection reason from debug info
                val debugInfo = criteriaResults.find { it.frameIndex == absoluteFrame }
                val reason = buildDetectionReason(debugInfo)

                shots.add(
                    ShotDetection(
                        frameIndex = i,
                        absoluteFrameIndex = absoluteFrame,
                        actionClasses = actions,
                        confidence = coarseScores[i][1],
                        detectionReason = reason
                    )
                )

//                Log.d(TAG, "✅ Shot at frame $absoluteFrame (conf: ${String.format("%.4f", coarseScores[i][1])})")
//                Log.d(TAG, "   Reason: $reason")
                if (actions.isNotEmpty()) {
//                    Log.d(TAG, "   Actions: ${actions.map { "${it.label}(${String.format("%.3f", it.confidence)})" }}")
                }
            }
        }

//        Log.d(TAG, "✅ RELAXED FINAL: ${shots.size} shots recorded")

        return F3SetResult(
            frameCoarsePredictions = coarsePredictions,
            frameCoarseScores = coarseScores,
            frameFinePredictions = finePredictions,
            shots = shots,
            inferenceTimeMs = 0,
            clipStartFrame = startFrameIndex,
            debugInfo = DebugInfo(fgCurve, criteriaResults)
        )
    }

    /**
     * Relaxed detection with lower thresholds and early exit
     */
    private fun detectShotsRelaxed(
        scores: Array<FloatArray>,
        startFrameIndex: Int
    ): Pair<IntArray, List<CriteriaDebug>> {
        val detections = IntArray(scores.size)
        val debugResults = mutableListOf<CriteriaDebug>()
        val window = 3

        for (i in scores.indices) {
            val fgScore = scores[i][1]
            val bgScore = scores[i][0]

            // EARLY EXIT: If fg > bg, immediately detect (shortcut for very strong signals)
            val earlyExit = ENABLE_EARLY_EXIT && fgScore > bgScore

            // Detection criteria (evaluate all for debugging)
            val strongDetection = ENABLE_STRONG && fgScore > bgScore && fgScore > STRONG_THRESHOLD
            val mediumDetection = ENABLE_MEDIUM && fgScore > MEDIUM_THRESHOLD

            // Local maxima detection
            var isLocalMaxima = false
            var peakStrength = 0f

            if (ENABLE_LOCAL_MAX && fgScore > LOCAL_MAX_MIN) {
                val start = maxOf(0, i - window)
                val end = minOf(scores.size - 1, i + window)

                isLocalMaxima = true
                var maxNeighbor = 0f

                for (j in start..end) {
                    if (j != i) {
                        maxNeighbor = maxOf(maxNeighbor, scores[j][1])
                        if (scores[j][1] >= fgScore) {
                            isLocalMaxima = false
                        }
                    }
                }

                if (isLocalMaxima) {
                    peakStrength = fgScore - maxNeighbor
                }
            }

            val localMaximaDetection = ENABLE_LOCAL_MAX && isLocalMaxima && peakStrength > PEAK_STRENGTH_MIN

            // Relative strength
            val relativeStrength = fgScore / (fgScore + bgScore)
            val relativeDetection = ENABLE_RELATIVE && relativeStrength > RELATIVE_THRESHOLD

            // Contextual detection
            var contextualDetection = false
            if (ENABLE_CONTEXTUAL && fgScore > CONTEXTUAL_MIN) {
                val contextStart = maxOf(0, i - 8)
                val contextEnd = minOf(scores.size - 1, i + 8)
                val contextScores = (contextStart..contextEnd).filter { it != i }.map { scores[it][1] }
                if (contextScores.isNotEmpty()) {
                    val contextMean = contextScores.average().toFloat()
                    contextualDetection = fgScore > (contextMean * CONTEXTUAL_MULTIPLIER)
                }
            }

            // Final decision: Keep if ANY criterion is met (OR logic)
            val keepDetection = earlyExit ||
                    strongDetection ||
                    mediumDetection ||
                    localMaximaDetection ||
                    relativeDetection ||
                    contextualDetection

            detections[i] = if (keepDetection) 1 else 0

            // Store debug info
            debugResults.add(
                CriteriaDebug(
                    frameIndex = startFrameIndex + i,
                    fgScore = fgScore,
                    bgScore = bgScore,
                    strong = strongDetection,
                    medium = mediumDetection,
                    localMax = localMaximaDetection,
                    relative = relativeDetection,
                    contextual = contextualDetection,
                    earlyExit = earlyExit,
                    finalDecision = keepDetection
                )
            )

            if (keepDetection) {
                val absoluteFrame = startFrameIndex + i
//                Log.d(TAG, "🎯 DETECTION frame $absoluteFrame: fg=${String.format("%.4f", fgScore)} " +
//                        "(early=$earlyExit, strong=$strongDetection, med=$mediumDetection, " +
//                        "localMax=$localMaximaDetection, rel=$relativeDetection, ctx=$contextualDetection)")
            }
        }

        return Pair(detections, debugResults)
    }

    private fun buildDetectionReason(debug: CriteriaDebug?): String {
        if (debug == null) return "unknown"

        val reasons = mutableListOf<String>()
        if (debug.earlyExit) reasons.add("early_exit(fg>bg)")
        if (debug.strong) reasons.add("strong(${String.format("%.3f", debug.fgScore)})")
        if (debug.medium) reasons.add("medium(${String.format("%.3f", debug.fgScore)})")
        if (debug.localMax) reasons.add("local_max")
        if (debug.relative) reasons.add("relative")
        if (debug.contextual) reasons.add("contextual")

        return if (reasons.isNotEmpty()) reasons.joinToString("+") else "none"
    }

    private fun applyTennisRules(finePreds: FloatArray): List<Detection> {
        val selected = mutableListOf<Detection>()
        val actionGroups = listOf(
            0 to 2, 2 to 5, 5 to 8, 16 to 24, 25 to 29
        )

        for ((start, end) in actionGroups) {
            var maxIdx = -1
            var maxScore = -1f
            for (i in start until minOf(end, finePreds.size)) {
                if (finePreds[i] > maxScore) {
                    maxScore = finePreds[i]
                    maxIdx = i
                }
            }
            if (maxIdx >= 0 && maxScore > 0.0f) {
                selected.add(Detection(getClassName(maxIdx), maxScore, null))
            }
        }

        if (24 < finePreds.size && finePreds[24] > 0.5f) {
            if (!selected.any { getClassIndex(it.label) == 24 }) {
                selected.add(Detection(getClassName(24), finePreds[24], null))
            }
        }

        val hasServe = selected.any { getClassIndex(it.label) == 5 }
        if (!hasServe) {
            val additionalGroups = listOf(
                8 to minOf(10, finePreds.size),
                10 to minOf(16, finePreds.size)
            )
            for ((start, end) in additionalGroups) {
                var maxIdx = -1
                var maxScore = -1f
                for (i in start until end) {
                    if (finePreds[i] > maxScore) {
                        maxScore = finePreds[i]
                        maxIdx = i
                    }
                }
                if (maxIdx >= 0 && maxScore > 0.0f) {
                    selected.add(Detection(getClassName(maxIdx), maxScore, null))
                }
            }
        }

        return selected.sortedByDescending { it.confidence }
    }

    private fun extractFloatData(tensor: Tensor): FloatArray {
        return try {
            tensor.dataAsFloatArray
        } catch (e: Exception) {
//            Log.e(TAG, "Failed to extract float data", e)
            floatArrayOf()
        }
    }

    private fun getClassName(index: Int): String {
        return tennisClasses[index] ?: "class_$index"
    }

    private fun getClassIndex(className: String): Int {
        return tennisClasses.entries.find { it.value == className }?.key ?: -1
    }

    private val tennisClasses = mapOf(
        0 to "near", 1 to "far", 2 to "deuce", 3 to "middle", 4 to "ad",
        5 to "serve", 6 to "return", 7 to "stroke", 8 to "fh", 9 to "bh",
        10 to "gs", 11 to "slice", 12 to "volley", 13 to "smash", 14 to "drop", 15 to "lob",
        16 to "T", 17 to "B", 18 to "W", 19 to "CC", 20 to "DL", 21 to "DM", 22 to "II", 23 to "IO",
        24 to "approach", 25 to "in", 26 to "winner", 27 to "forced-err", 28 to "unforced-err"
    )

    private fun copyAssetToFile(assetName: String): String {
        val fileName = assetName.substringAfterLast("/")
        val file = File(context.filesDir, fileName)

        if (!file.exists()) {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        return file.absolutePath
    }

    fun release() {
        module = null
        isModelLoaded = false
        System.gc()
    }
}