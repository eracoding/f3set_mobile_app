package com.example.aivideoanalyzer.presentation.ui.processing

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.aivideoanalyzer.R
import com.example.aivideoanalyzer.databinding.ItemProcessingVideoBinding
import com.example.aivideoanalyzer.domain.model.Video
import com.example.aivideoanalyzer.domain.model.VideoStatus
import com.google.android.material.color.MaterialColors

class ProcessingVideoAdapter(
    private val onVideoClick: (Video) -> Unit,
    private val onActionClick: (Video) -> Unit
) : ListAdapter<Video, ProcessingVideoAdapter.VideoViewHolder>(VideoDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VideoViewHolder {
        val binding = ItemProcessingVideoBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return VideoViewHolder(binding)
    }

    override fun onBindViewHolder(holder: VideoViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class VideoViewHolder(
        private val binding: ItemProcessingVideoBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        init {
            binding.root.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onVideoClick(getItem(position))
                }
            }

            binding.actionButton.setOnClickListener {
                val position = adapterPosition
                if (position != RecyclerView.NO_POSITION) {
                    onActionClick(getItem(position))
                }
            }
        }

        fun bind(video: Video) {
            binding.videoName.text = video.name
            binding.videoInfo.text = formatVideoInfo(video)

            // Update status chip
            binding.statusChip.text = getStatusText(video.status)
            binding.statusChip.chipBackgroundColor = getStatusColor(video.status)

            // Update progress visibility and value
            when (video.status) {
                VideoStatus.PREPROCESSING, VideoStatus.PROCESSING -> {
                    binding.progressIndicator.visibility = View.VISIBLE
                    binding.progressText.visibility = View.VISIBLE

                    // Get actual progress from global processing state
                    val progress = getActualProgress(video)
                    binding.progressIndicator.progress = progress
                    binding.progressText.text = "$progress%"

                    // Make progress bar indeterminate if we don't have real progress
                    binding.progressIndicator.isIndeterminate = (progress == 0)
                }
                else -> {
                    binding.progressIndicator.visibility = View.GONE
                    binding.progressText.visibility = View.GONE
                    binding.progressIndicator.isIndeterminate = false
                }
            }

            // Update action button
            binding.actionButton.setIconResource(getActionIcon(video.status))
            binding.actionButton.isEnabled = video.status != VideoStatus.PROCESSING
        }

        private fun getActualProgress(video: Video): Int {
            // This is a placeholder - you might want to store actual progress in the Video model
            // or get it from a global processing state manager
            return when (video.status) {
                VideoStatus.PREPROCESSING -> 25
                VideoStatus.PROCESSING -> 65
                else -> 0
            }
        }

        private fun formatVideoInfo(video: Video): String {
            val sizeInMB = video.size / (1024.0 * 1024.0)
            val duration = formatDuration(video.duration)
            return String.format("%.1f MB • %s", sizeInMB, duration)
        }

        private fun formatDuration(milliseconds: Long): String {
            val seconds = milliseconds / 1000
            val minutes = seconds / 60
            val remainingSeconds = seconds % 60
            return String.format("%d:%02d", minutes, remainingSeconds)
        }

        private fun getStatusText(status: VideoStatus): String {
            return when (status) {
                VideoStatus.UPLOADED -> "Ready for F3Set Analysis"
                VideoStatus.PREPROCESSING -> "Preparing for F3Set"
                VideoStatus.PROCESSING -> "F3Set Analysis Running"
                VideoStatus.COMPLETED -> "Analysis Complete"
                VideoStatus.ERROR -> "Analysis Failed"
            }
        }

        private fun getStatusColor(status: VideoStatus): android.content.res.ColorStateList? {
            val context = binding.root.context
            val color = when (status) {
                VideoStatus.UPLOADED -> MaterialColors.getColor(binding.root, com.google.android.material.R.attr.colorSurfaceVariant)
                VideoStatus.PREPROCESSING, VideoStatus.PROCESSING -> context.getColor(R.color.video_processing)
                VideoStatus.COMPLETED -> context.getColor(R.color.video_completed)
                VideoStatus.ERROR -> context.getColor(R.color.video_error)
            }
            return android.content.res.ColorStateList.valueOf(color)
        }

        private fun getActionIcon(status: VideoStatus): Int {
            return when (status) {
                VideoStatus.UPLOADED -> R.drawable.ic_play // Start F3Set analysis
                VideoStatus.PREPROCESSING, VideoStatus.PROCESSING -> R.drawable.ic_pause // Pause
                VideoStatus.COMPLETED -> R.drawable.ic_check // View results
                VideoStatus.ERROR -> R.drawable.ic_refresh // Retry
            }
        }
    }

    class VideoDiffCallback : DiffUtil.ItemCallback<Video>() {
        override fun areItemsTheSame(oldItem: Video, newItem: Video): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: Video, newItem: Video): Boolean {
            return oldItem == newItem
        }
    }
}