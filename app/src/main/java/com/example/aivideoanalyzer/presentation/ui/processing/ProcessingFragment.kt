package com.example.aivideoanalyzer.presentation.ui.processing

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aivideoanalyzer.databinding.FragmentProcessingBinding
import com.example.aivideoanalyzer.presentation.ui.ViewModelFactory
import com.google.android.material.chip.Chip

class ProcessingFragment : Fragment() {

    private var _binding: FragmentProcessingBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ProcessingViewModel by viewModels {
        ViewModelFactory(requireActivity().application)
    }

    private lateinit var videoAdapter: ProcessingVideoAdapter

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentProcessingBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupRecyclerView()
        setupFilters()
        observeViewModel()
    }

    private fun setupRecyclerView() {
        videoAdapter = ProcessingVideoAdapter(
            onVideoClick = { video ->
                viewModel.onVideoSelected(video)
            },
            onActionClick = { video ->
                viewModel.toggleProcessing(video)
            }
        )

        binding.videosRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = videoAdapter
        }
    }

    private fun setupFilters() {
        val statuses = listOf("All", "Processing", "Completed", "Error")

        statuses.forEach { status ->
            val chip = Chip(context).apply {
                text = status
                isCheckable = true
                isChecked = status == "All"

                setOnCheckedChangeListener { _, isChecked ->
                    if (isChecked) {
                        viewModel.filterByStatus(status)
                    }
                }
            }
            binding.filterChipGroup.addView(chip)
        }
    }

    private fun observeViewModel() {
        viewModel.processingVideos.observe(viewLifecycleOwner) { videos ->
            videoAdapter.submitList(videos)
            updateEmptyState(videos.isEmpty())
        }

        viewModel.processingState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is ProcessingViewModel.ProcessingState.Idle -> {
                    binding.globalProgressCard.visibility = View.GONE
                }
                is ProcessingViewModel.ProcessingState.Processing -> {
                    binding.globalProgressCard.visibility = View.VISIBLE
                    binding.currentProcessingText.text = state.videoName
                    binding.processingProgressBar.progress = state.progress
                    binding.processingProgressText.text = "${state.progress}%"
                }
                is ProcessingViewModel.ProcessingState.Completed -> {
                    binding.globalProgressCard.visibility = View.GONE
                    // Show success message
                }
                is ProcessingViewModel.ProcessingState.Error -> {
                    binding.globalProgressCard.visibility = View.GONE
                    // Show error message
                }
            }
        }

        viewModel.statistics.observe(viewLifecycleOwner) { stats ->
            binding.totalVideosText.text = "Total: ${stats.total}"
            binding.processingCountText.text = "Processing: ${stats.processing}"
            binding.completedCountText.text = "Completed: ${stats.completed}"
        }
    }

    private fun updateEmptyState(isEmpty: Boolean) {
        if (isEmpty) {
            binding.emptyStateLayout.visibility = View.VISIBLE
            binding.videosRecyclerView.visibility = View.GONE
        } else {
            binding.emptyStateLayout.visibility = View.GONE
            binding.videosRecyclerView.visibility = View.VISIBLE
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}