package com.example.aivideoanalyzer.presentation.ui.results

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aivideoanalyzer.databinding.FragmentResultsBinding
import com.example.aivideoanalyzer.domain.model.AnalysisResult
import com.example.aivideoanalyzer.presentation.ui.ViewModelFactory
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar

class ResultsFragment : Fragment() {

    private var _binding: FragmentResultsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ResultsViewModel by viewModels { ViewModelFactory(requireActivity().application) }
    private lateinit var resultsAdapter: ResultsAdapter

    // For handling save file dialog
    private val saveFileLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == android.app.Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                currentSaveResult?.let { analysisResult ->
                    currentSaveFormat?.let { format ->
                        viewModel.saveFileToUri(analysisResult, format, uri)
                    }
                }
            }
        }
        currentSaveResult = null
        currentSaveFormat = null
    }

    // Temporary variables for save operation
    private var currentSaveResult: AnalysisResult? = null
    private var currentSaveFormat: String? = null

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentResultsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupRecyclerView()
        setupUI()
        observeViewModel()
    }

    private fun setupRecyclerView() {
        resultsAdapter = ResultsAdapter(
            onResultClick = { result ->
                showDetailDialog(result)
            },
            onExportClick = { result ->
                showExportOptionsDialog(result)
            },
            onShareClick = { result ->
                showShareOptionsDialog(result)
            }
        )

        binding.resultsRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = resultsAdapter
        }
    }

    private fun setupUI() {
        binding.swipeRefreshLayout.setOnRefreshListener {
            viewModel.refreshResults()
        }

        binding.searchView.setOnQueryTextListener(object : android.widget.SearchView.OnQueryTextListener {
            override fun onQueryTextSubmit(query: String?): Boolean {
                return false
            }

            override fun onQueryTextChange(newText: String?): Boolean {
                viewModel.searchResults(newText ?: "")
                return true
            }
        })

        binding.sortButton.setOnClickListener {
            showSortDialog()
        }

        binding.exportAllButton.setOnClickListener {
            viewModel.exportAllResults()
        }
    }

    private fun observeViewModel() {
        viewModel.analysisResults.observe(viewLifecycleOwner) { results ->
            resultsAdapter.submitList(results)
            updateEmptyState(results.isEmpty())
            binding.swipeRefreshLayout.isRefreshing = false
        }

        viewModel.exportState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is ResultsViewModel.ExportState.Idle -> {
                    binding.exportProgressBar.visibility = View.GONE
                }
                is ResultsViewModel.ExportState.Exporting -> {
                    binding.exportProgressBar.visibility = View.VISIBLE
                }
                is ResultsViewModel.ExportState.Success -> {
                    binding.exportProgressBar.visibility = View.GONE
                    showMessage("Export completed successfully")
                }
                is ResultsViewModel.ExportState.Error -> {
                    binding.exportProgressBar.visibility = View.GONE
                    showError(state.message)
                }
            }
        }

        viewModel.shareState.observe(viewLifecycleOwner) { state ->
            when (state) {
                is ResultsViewModel.ShareState.Idle -> {
                    // Do nothing
                }
                is ResultsViewModel.ShareState.ReadyToShare -> {
                    startActivity(Intent.createChooser(state.shareIntent, "Share Analysis"))
                    viewModel.clearShareState()
                }
                is ResultsViewModel.ShareState.ReadyToSave -> {
                    saveFileLauncher.launch(state.saveIntent)
                    viewModel.clearShareState()
                }
                is ResultsViewModel.ShareState.Error -> {
                    showError(state.message)
                    viewModel.clearShareState()
                }
            }
        }
    }

    private fun showDetailDialog(result: AnalysisResult) {
        val dialogView = layoutInflater.inflate(
            com.example.aivideoanalyzer.R.layout.dialog_analysis_detail,
            null
        )

        // Get references to dialog views
        val videoNameText = dialogView.findViewById<android.widget.TextView>(
            com.example.aivideoanalyzer.R.id.detail_video_name
        )
        val timestampText = dialogView.findViewById<android.widget.TextView>(
            com.example.aivideoanalyzer.R.id.detail_timestamp
        )
        val confidenceText = dialogView.findViewById<android.widget.TextView>(
            com.example.aivideoanalyzer.R.id.detail_confidence
        )
        val summaryText = dialogView.findViewById<android.widget.TextView>(
            com.example.aivideoanalyzer.R.id.detail_summary
        )
        val framesRecyclerView = dialogView.findViewById<androidx.recyclerview.widget.RecyclerView>(
            com.example.aivideoanalyzer.R.id.frames_recycler_view
        )
        val statsText = dialogView.findViewById<android.widget.TextView>(
            com.example.aivideoanalyzer.R.id.detail_stats
        )

        // Populate dialog content with corrected confidence calculation
        videoNameText.text = "🎾 ${result.videoId}"
        timestampText.text = "📅 Analysis completed: ${android.text.format.DateFormat.format("MMM dd, yyyy HH:mm", result.timestamp)}"

        // Calculate proper confidence from detection confidences
        val detectionConfidences = result.frames.flatMap { it.detections }.map { it.confidence }
        val calculatedConfidence = if (detectionConfidences.isNotEmpty()) {
            detectionConfidences.average().toFloat()
        } else {
            result.confidence
        }

        confidenceText.text = "🎯 Overall Confidence: ${String.format("%.1f%%", calculatedConfidence * 100)}"
        summaryText.text = result.summary

        // Setup frame details RecyclerView
        val frameAdapter = FrameDetailsAdapter()
        framesRecyclerView.apply {
            layoutManager = LinearLayoutManager(context)
            adapter = frameAdapter
        }
        frameAdapter.submitList(result.frames)

        // Calculate and show statistics with corrected counts
        val totalShots = result.frames.count { it.detections.isNotEmpty() }
        val totalDetections = result.frames.sumOf { it.detections.size }
        val uniqueActions = result.frames.flatMap { it.detections }
            .map { it.label }
            .distinct()
            .size
        val avgFrameConfidence = if (result.frames.isNotEmpty()) {
            result.frames.map { it.confidence }.average()
        } else 0.0

        statsText.text = buildString {
            appendLine("📊 Analysis Statistics:")
            appendLine("• Frames analyzed: ${result.frames.size}")
            appendLine("• Tennis shots detected: $totalShots")
            appendLine("• Individual detections: $totalDetections")
            appendLine("• Unique action types: $uniqueActions")
            appendLine("• Average frame confidence: ${String.format("%.2f%%", avgFrameConfidence * 100)}")
            appendLine("• Calculated overall confidence: ${String.format("%.2f%%", calculatedConfidence * 100)}")

            if (result.frames.isNotEmpty()) {
                appendLine("\n🏆 Top detected actions:")
                val topActions = result.frames.flatMap { it.detections }
                    .groupBy { it.label }
                    .mapValues { it.value.size }
                    .toList()
                    .sortedByDescending { it.second }
                    .take(5)

                topActions.forEach { (action, count) ->
                    appendLine("• $action: $count occurrence(s)")
                }
            }
        }

        // Create and show dialog
        MaterialAlertDialogBuilder(requireContext())
            .setTitle("F3Set Analysis Details")
            .setView(dialogView)
            .setPositiveButton("Close") { dialog, _ ->
                dialog.dismiss()
            }
            .setNeutralButton("Share") { _, _ ->
                showShareOptionsDialog(result)
            }
            .setNegativeButton("Export") { _, _ ->
                showExportOptionsDialog(result)
            }
            .create()
            .show()
    }

    private fun showExportOptionsDialog(result: AnalysisResult) {
        val options = arrayOf("HTML Report", "CSV Data", "JSON Data")

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("📤 Export F3Set Analysis")
            .setIcon(com.example.aivideoanalyzer.R.drawable.ic_export)
            .setItems(options) { _, which ->
                val format = when (which) {
                    0 -> "html"
                    1 -> "csv"
                    2 -> "json"
                    else -> "html"
                }

                // Show save or share options after selecting format
                showSaveOrShareDialog(result, format)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showSaveOrShareDialog(result: AnalysisResult, format: String) {
        val formatName = when (format) {
            "html" -> "HTML Report"
            "csv" -> "CSV Data"
            "json" -> "JSON Data"
            else -> "File"
        }

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Save or Share $formatName")
            .setMessage("Choose how you want to handle the exported file:")
            .setPositiveButton("💾 Save to Device") { _, _ ->
                currentSaveResult = result
                currentSaveFormat = format
                viewModel.prepareSaveIntent(result, format)
            }
            .setNegativeButton("📤 Share") { _, _ ->
                viewModel.exportResult(result, format)
            }
            .setNeutralButton("Cancel", null)
            .show()
    }

    private fun showShareOptionsDialog(result: AnalysisResult) {
        val options = arrayOf(
            "📱 Share as Text Message",
            "📄 Share HTML Report",
            "📊 Share CSV Data",
            "🔧 Share JSON Data"
        )

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("🔗 Share Analysis")
            .setIcon(com.example.aivideoanalyzer.R.drawable.ic_share)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> viewModel.shareResultAsText(result)
                    1 -> viewModel.exportResult(result, "html")
                    2 -> viewModel.exportResult(result, "csv")
                    3 -> viewModel.exportResult(result, "json")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showSortDialog() {
        val options = arrayOf("Date (Newest)", "Date (Oldest)", "Confidence (High)", "Confidence (Low)")

        MaterialAlertDialogBuilder(requireContext())
            .setTitle("Sort Analysis Results")
            .setItems(options) { _, which ->
                viewModel.sortResults(which)
            }
            .show()
    }

    private fun updateEmptyState(isEmpty: Boolean) {
        if (isEmpty) {
            binding.emptyStateLayout.visibility = View.VISIBLE
            binding.resultsRecyclerView.visibility = View.GONE
        } else {
            binding.emptyStateLayout.visibility = View.GONE
            binding.resultsRecyclerView.visibility = View.VISIBLE
        }
    }

    private fun showMessage(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
            .setAction("Dismiss") { }
            .show()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}