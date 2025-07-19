package com.example.aivideoanalyzer.presentation.ui.results

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.aivideoanalyzer.databinding.ItemFrameDetailBinding
import com.example.aivideoanalyzer.domain.model.FrameAnalysis

class FrameDetailsAdapter : ListAdapter<FrameAnalysis, FrameDetailsAdapter.FrameViewHolder>(FrameDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FrameViewHolder {
        val binding = ItemFrameDetailBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return FrameViewHolder(binding)
    }

    override fun onBindViewHolder(holder: FrameViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class FrameViewHolder(
        private val binding: ItemFrameDetailBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(frame: FrameAnalysis) {
            // Frame info
            binding.frameNumberText.text = "Frame ${frame.frameNumber}"
            binding.timestampText.text = formatTimestamp(frame.timestamp)
            binding.frameConfidenceText.text = "Confidence: ${String.format("%.2f%%", frame.confidence * 100)}"

            // Detections
            if (frame.detections.isNotEmpty()) {
                val detectionsText = buildString {
                    frame.detections.sortedByDescending { it.confidence }.forEach { detection ->
                        appendLine("🎾 ${detection.label}")
                        appendLine("   Confidence: ${String.format("%.2f%%", detection.confidence * 100)}")
                        detection.boundingBox?.let { bbox ->
                            appendLine("   Position: x=${String.format("%.0f", bbox.x)}, y=${String.format("%.0f", bbox.y)}")
                            appendLine("   Size: ${String.format("%.0f", bbox.width)}×${String.format("%.0f", bbox.height)}")
                        }
                        appendLine()
                    }
                }
                binding.detectionsText.text = detectionsText.trim()
            } else {
                binding.detectionsText.text = "No tennis actions detected in this frame"
            }

            // Visual styling based on detection count
            val detectionCount = frame.detections.size
            when {
                detectionCount == 0 -> {
                    binding.root.alpha = 0.7f
                    binding.frameNumberText.setTextColor(
                        binding.root.context.getColor(com.example.aivideoanalyzer.R.color.md_theme_onSurfaceVariant)
                    )
                }
                detectionCount >= 3 -> {
                    binding.root.alpha = 1.0f
                    binding.frameNumberText.setTextColor(
                        binding.root.context.getColor(com.example.aivideoanalyzer.R.color.video_completed)
                    )
                }
                else -> {
                    binding.root.alpha = 1.0f
                    binding.frameNumberText.setTextColor(
                        binding.root.context.getColor(com.example.aivideoanalyzer.R.color.video_processing)
                    )
                }
            }
        }

        private fun formatTimestamp(timestamp: Long): String {
            val seconds = timestamp / 1000
            val minutes = seconds / 60
            val remainingSeconds = seconds % 60
            val milliseconds = timestamp % 1000
            return String.format("%d:%02d.%03d", minutes, remainingSeconds, milliseconds)
        }
    }

    class FrameDiffCallback : DiffUtil.ItemCallback<FrameAnalysis>() {
        override fun areItemsTheSame(oldItem: FrameAnalysis, newItem: FrameAnalysis): Boolean {
            return oldItem.frameNumber == newItem.frameNumber &&
                    oldItem.timestamp == newItem.timestamp
        }

        override fun areContentsTheSame(oldItem: FrameAnalysis, newItem: FrameAnalysis): Boolean {
            return oldItem == newItem
        }
    }
}