package com.example.aivideoanalyzer.presentation.ui.results

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.aivideoanalyzer.databinding.ItemAnalysisResultBinding
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import java.text.SimpleDateFormat
import java.util.*

class ResultsAdapter(
    private val onResultClick: (AnalysisResult) -> Unit,
    private val onExportClick: (AnalysisResult) -> Unit,
    private val onShareClick: (AnalysisResult) -> Unit
) : ListAdapter<AnalysisResult, ResultsAdapter.ResultViewHolder>(ResultDiffCallback()) {

    private val dateFormat = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ResultViewHolder {
        val binding = ItemAnalysisResultBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ResultViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ResultViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ResultViewHolder(
        private val binding: ItemAnalysisResultBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        init {
            binding.root.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onResultClick(getItem(position))
                }
            }

            binding.exportButton.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onExportClick(getItem(position))
                }
            }

            binding.shareButton.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onShareClick(getItem(position))
                }
            }
        }

        fun bind(result: AnalysisResult) {
            // Enhanced video ID display with F3Set branding
            binding.videoIdText.text = "🎾 F3Set Analysis: ${result.videoId}"
            binding.timestampText.text = "📅 ${dateFormat.format(result.timestamp)}"

            // Calculate proper confidence from detection confidences
            val detectionConfidences = result.frames.flatMap { it.detections }.map { it.confidence }
            val calculatedConfidence = if (detectionConfidences.isNotEmpty()) {
                detectionConfidences.average().toFloat()
            } else {
                result.confidence
            }

            // Enhanced summary with F3Set context and corrected statistics
            val enhancedSummary = buildString {
                append("F3Set Tennis Analysis Results: ")

                // Get key statistics for quick preview (corrected counts)
                val totalShots = result.frames.count { it.detections.isNotEmpty() }
                val topActions = result.frames.flatMap { it.detections }
                    .groupBy { it.label }
                    .mapValues { it.value.size }
                    .toList()
                    .sortedByDescending { it.second }
                    .take(3)
                    .map { "${it.first} (${it.second})" }
                    .joinToString(", ")

                if (totalShots > 0) {
                    append("Detected $totalShots tennis shots")
                    if (topActions.isNotEmpty()) {
                        append(" including $topActions")
                    }
                    append(". ")
                }

                // Add original summary
                append(result.summary.take(100))
                if (result.summary.length > 100) {
                    append("...")
                }
            }
            binding.summaryText.text = enhancedSummary

            // Format confidence with corrected calculation
            val confidencePercent = (calculatedConfidence * 100).toInt()
            binding.confidenceText.text = "$confidencePercent%"

            // Set confidence indicator color with tennis theme
            val confidenceColor = getConfidenceColor(calculatedConfidence)
            binding.confidenceIndicator.setIndicatorColor(confidenceColor)
            binding.confidenceIndicator.progress = confidencePercent

            // Enhanced frame and detection count with tennis terminology (corrected)
            val frameCount = result.frames.size
            val shotDetections = result.frames.count { it.detections.isNotEmpty() }
            val uniqueActions = result.frames.flatMap { it.detections }
                .map { it.label }
                .distinct()
                .size

            binding.statsText.text = buildString {
                append("🎬 $frameCount frames analyzed")
                if (shotDetections > 0) {
                    append(" • 🎾 $shotDetections shots detected")
                    if (uniqueActions > 0) {
                        append(" • 🎯 $uniqueActions action types")
                    }
                } else {
                    append(" • No tennis shots detected")
                }
            }

            // Add visual enhancement for successful analysis
            if (shotDetections > 0) {
                binding.root.alpha = 1.0f
                // Add subtle success indicator
                binding.videoIdText.setCompoundDrawablesWithIntrinsicBounds(
                    com.example.aivideoanalyzer.R.drawable.ic_check, 0, 0, 0
                )
                binding.videoIdText.compoundDrawablePadding = 8
            } else {
                binding.root.alpha = 0.8f
                binding.videoIdText.setCompoundDrawablesWithIntrinsicBounds(0, 0, 0, 0)
            }

            // Style buttons with improved accessibility
            binding.exportButton.contentDescription = "Export analysis for ${result.videoId}"
            binding.shareButton.contentDescription = "Share analysis for ${result.videoId}"
        }

        private fun getConfidenceColor(confidence: Float): Int {
            val context = binding.root.context
            return when {
                confidence >= 0.8f -> context.getColor(com.example.aivideoanalyzer.R.color.video_completed)
                confidence >= 0.6f -> context.getColor(com.example.aivideoanalyzer.R.color.video_processing)
                else -> context.getColor(com.example.aivideoanalyzer.R.color.video_error)
            }
        }
    }

    class ResultDiffCallback : DiffUtil.ItemCallback<AnalysisResult>() {
        override fun areItemsTheSame(oldItem: AnalysisResult, newItem: AnalysisResult): Boolean {
            return oldItem.videoId == newItem.videoId &&
                    oldItem.timestamp == newItem.timestamp
        }

        override fun areContentsTheSame(oldItem: AnalysisResult, newItem: AnalysisResult): Boolean {
            return oldItem == newItem
        }
    }
}