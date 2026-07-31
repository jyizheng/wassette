// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Report generation for comparison benchmarks.

use std::fmt::Write;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::config::ComparisonConfig;
use crate::metrics::{ComparisonMetrics, ComparisonRow};

/// Report format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Markdown format.
    Markdown,
    /// JSON format.
    Json,
    /// CSV format.
    Csv,
    /// LaTeX table format.
    Latex,
}

impl Default for ReportFormat {
    fn default() -> Self {
        Self::Markdown
    }
}

impl ReportFormat {
    /// File extension for this report format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Latex => "tex",
        }
    }
}

/// Report generator for comparison results.
pub struct ReportGenerator {
    /// Configuration used.
    config: ComparisonConfig,

    /// Metrics to report on.
    metrics: ComparisonMetrics,

    /// Output format.
    format: ReportFormat,
}

impl ReportGenerator {
    /// Create a new report generator.
    pub fn new(config: ComparisonConfig, metrics: ComparisonMetrics) -> Self {
        Self {
            config,
            metrics,
            format: ReportFormat::Markdown,
        }
    }

    /// Set output format.
    pub fn with_format(mut self, format: ReportFormat) -> Self {
        self.format = format;
        self
    }

    /// Generate the report.
    pub fn generate(&self) -> Result<String> {
        match self.format {
            ReportFormat::Markdown => self.generate_markdown(),
            ReportFormat::Json => self.generate_json(),
            ReportFormat::Csv => self.generate_csv(),
            ReportFormat::Latex => self.generate_latex(),
        }
    }

    /// Generate markdown report.
    fn generate_markdown(&self) -> Result<String> {
        let mut output = String::new();

        // Header
        writeln!(output, "# WasmRL vs CAMEL-AI Comparison Report")?;
        writeln!(output)?;

        // Configuration summary
        writeln!(output, "## Configuration")?;
        writeln!(output)?;
        writeln!(
            output,
            "- **Benchmark iterations**: {}",
            self.config.iterations
        )?;
        writeln!(output, "- **Warmup iterations**: {}", self.config.warmup)?;
        writeln!(
            output,
            "- **Backends tested**: {}",
            self.config.backends.len()
        )?;
        writeln!(output)?;

        // Hardware info
        let hw = &self.config.hardware;
        writeln!(output, "## Hardware")?;
        writeln!(output)?;
        writeln!(output, "- **CPU**: {} cores", hw.cpu_cores)?;
        writeln!(output, "- **Memory**: {} GB", hw.memory_gb)?;
        if let Some(os) = &hw.os_info {
            writeln!(output, "- **OS**: {}", os)?;
        }
        writeln!(output)?;

        // Main comparison table
        writeln!(output, "## Performance Comparison")?;
        writeln!(output)?;
        self.write_markdown_table(&mut output)?;
        writeln!(output)?;

        // Scaling results
        if let Some(scaling) = &self.metrics.scaling {
            writeln!(output, "## Scaling Analysis")?;
            writeln!(output)?;
            writeln!(output, "Backend: **{}**", scaling.backend)?;
            writeln!(output)?;
            writeln!(
                output,
                "| Env Count | Throughput (sps) | P99 Latency (µs) |"
            )?;
            writeln!(output, "|-----------|-----------------|------------------|")?;
            for point in &scaling.results {
                writeln!(
                    output,
                    "| {} | {:.0} | {} |",
                    point.env_count, point.throughput_sps, point.latency_p99_us
                )?;
            }
            writeln!(output)?;
            let linear = if scaling.is_linear(0.2) { "Yes" } else { "No" };
            writeln!(output, "Linear scaling (within 20%): **{}**", linear)?;
            writeln!(output)?;
        }

        // Cold-start analysis
        if let Some(cold) = &self.metrics.cold_start {
            writeln!(output, "## Cold-Start Analysis")?;
            writeln!(output)?;
            writeln!(output, "Backend: **{}**", cold.backend)?;
            writeln!(output)?;
            writeln!(output, "| Metric | Value (µs) |")?;
            writeln!(output, "|--------|-----------|")?;
            writeln!(output, "| First step (cold) | {} |", cold.first_step_us)?;
            writeln!(output, "| Warm step | {} |", cold.warm_step_us)?;
            writeln!(
                output,
                "| Instance creation | {} |",
                cold.instance_create_us
            )?;
            writeln!(output, "| Overhead ratio | {:.1}x |", cold.overhead_ratio())?;
            writeln!(output)?;
        }

        // Methodology
        writeln!(output, "## Methodology")?;
        writeln!(output)?;
        writeln!(
            output,
            "All measurements follow the fairness protocol defined in CAMEL_COMPARISON.md:"
        )?;
        writeln!(output)?;
        writeln!(output, "1. Same tasks executed across all backends")?;
        writeln!(output, "2. Warmup phase before measurement")?;
        writeln!(output, "3. P99 latency used for tail latency comparison")?;
        writeln!(output, "4. Cold-start measured with fresh instances")?;
        writeln!(output)?;

        Ok(output)
    }

    /// Write markdown comparison table.
    fn write_markdown_table(&self, output: &mut String) -> Result<()> {
        let rows = self.metrics.comparison_table();

        writeln!(output, "| Backend | Step Mean (µs) | Step P99 (µs) | Reset Mean (µs) | Throughput (sps) | Speedup |")?;
        writeln!(output, "|---------|---------------|--------------|-----------------|-----------------|---------|")?;

        for row in rows {
            writeln!(
                output,
                "| {} | {} | {} | {} | {:.0} | {:.1}x |",
                row.backend,
                row.step_mean_us,
                row.step_p99_us,
                row.reset_mean_us,
                row.throughput_sps,
                row.speedup_vs_baseline
            )?;
        }

        Ok(())
    }

    /// Generate JSON report.
    fn generate_json(&self) -> Result<String> {
        let report = JsonReport {
            config_summary: ConfigSummary {
                benchmark_iterations: self.config.iterations,
                warmup_iterations: self.config.warmup,
                backends: self.config.backends.iter().map(|b| b.to_string()).collect(),
            },
            comparison_table: self.metrics.comparison_table(),
            scaling: self.metrics.scaling.as_ref().map(|s| s.results.clone()),
            cold_start: self.metrics.cold_start.as_ref().map(|c| ColdStartSummary {
                backend: c.backend.to_string(),
                first_step_us: c.first_step_us,
                warm_step_us: c.warm_step_us,
                overhead_ratio: c.overhead_ratio(),
            }),
        };

        Ok(serde_json::to_string_pretty(&report)?)
    }

    /// Generate CSV report.
    fn generate_csv(&self) -> Result<String> {
        let mut output = String::new();

        writeln!(
            output,
            "backend,step_mean_us,step_p99_us,reset_mean_us,throughput_sps,speedup"
        )?;

        for row in self.metrics.comparison_table() {
            writeln!(
                output,
                "{},{},{},{},{:.0},{:.2}",
                row.backend,
                row.step_mean_us,
                row.step_p99_us,
                row.reset_mean_us,
                row.throughput_sps,
                row.speedup_vs_baseline
            )?;
        }

        Ok(output)
    }

    /// Generate LaTeX table.
    fn generate_latex(&self) -> Result<String> {
        let mut output = String::new();

        writeln!(output, "\\begin{{table}}[htbp]")?;
        writeln!(output, "\\centering")?;
        writeln!(
            output,
            "\\caption{{WasmRL vs CAMEL-AI Performance Comparison}}"
        )?;
        writeln!(output, "\\label{{tab:comparison}}")?;
        writeln!(output, "\\begin{{tabular}}{{lrrrrr}}")?;
        writeln!(output, "\\toprule")?;
        writeln!(output, "Backend & Step Mean ($\\mu$s) & Step P99 ($\\mu$s) & Reset ($\\mu$s) & Throughput & Speedup \\\\")?;
        writeln!(output, "\\midrule")?;

        for row in self.metrics.comparison_table() {
            writeln!(
                output,
                "{} & {} & {} & {} & {:.0} & {:.1}$\\times$ \\\\",
                row.backend.replace("_", "\\_"),
                row.step_mean_us,
                row.step_p99_us,
                row.reset_mean_us,
                row.throughput_sps,
                row.speedup_vs_baseline
            )?;
        }

        writeln!(output, "\\bottomrule")?;
        writeln!(output, "\\end{{tabular}}")?;
        writeln!(output, "\\end{{table}}")?;

        Ok(output)
    }

    /// Save report to file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let content = self.generate()?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// JSON report structure.
#[derive(Debug, Serialize, Deserialize)]
struct JsonReport {
    config_summary: ConfigSummary,
    comparison_table: Vec<ComparisonRow>,
    scaling: Option<Vec<crate::metrics::ScalingPoint>>,
    cold_start: Option<ColdStartSummary>,
}

/// Configuration summary for JSON.
#[derive(Debug, Serialize, Deserialize)]
struct ConfigSummary {
    benchmark_iterations: usize,
    warmup_iterations: usize,
    backends: Vec<String>,
}

/// Cold-start summary for JSON.
#[derive(Debug, Serialize, Deserialize)]
struct ColdStartSummary {
    backend: String,
    first_step_us: u64,
    warm_step_us: u64,
    overhead_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::metrics::BackendMetrics;

    fn sample_metrics() -> ComparisonMetrics {
        let mut metrics = ComparisonMetrics::new();

        let mut wasm = BackendMetrics::new(Backend::WasmInproc);
        wasm.step_mean_us = 100;
        wasm.step_p99_us = 200;
        wasm.reset_mean_us = 50;
        wasm.throughput_steps_per_sec = 10000.0;
        metrics.add_backend(wasm);

        let mut docker = BackendMetrics::new(Backend::DockerTask);
        docker.step_mean_us = 5000;
        docker.step_p99_us = 10000;
        docker.reset_mean_us = 2000;
        docker.throughput_steps_per_sec = 200.0;
        metrics.add_backend(docker);

        metrics
    }

    #[test]
    fn test_report_format_default() {
        assert_eq!(ReportFormat::default(), ReportFormat::Markdown);
    }

    #[test]
    fn test_generate_markdown() {
        let config = ComparisonConfig::minimal();
        let metrics = sample_metrics();
        let generator = ReportGenerator::new(config, metrics);

        let report = generator.generate().unwrap();
        assert!(report.contains("# WasmRL vs CAMEL-AI Comparison Report"));
        assert!(report.contains("Performance Comparison"));
        assert!(report.contains("WasmInproc"));
    }

    #[test]
    fn test_generate_json() {
        let config = ComparisonConfig::minimal();
        let metrics = sample_metrics();
        let generator = ReportGenerator::new(config, metrics).with_format(ReportFormat::Json);

        let report = generator.generate().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&report).unwrap();
        assert!(parsed.get("config_summary").is_some());
        assert!(parsed.get("comparison_table").is_some());
    }

    #[test]
    fn test_generate_csv() {
        let config = ComparisonConfig::minimal();
        let metrics = sample_metrics();
        let generator = ReportGenerator::new(config, metrics).with_format(ReportFormat::Csv);

        let report = generator.generate().unwrap();
        assert!(report.contains("backend,step_mean_us"));
        assert!(report.lines().count() >= 2);
    }

    #[test]
    fn test_generate_latex() {
        let config = ComparisonConfig::minimal();
        let metrics = sample_metrics();
        let generator = ReportGenerator::new(config, metrics).with_format(ReportFormat::Latex);

        let report = generator.generate().unwrap();
        assert!(report.contains("\\begin{table}"));
        assert!(report.contains("\\end{table}"));
        assert!(report.contains("\\toprule"));
    }

    #[test]
    fn test_with_format() {
        let config = ComparisonConfig::minimal();
        let metrics = sample_metrics();
        let generator = ReportGenerator::new(config, metrics).with_format(ReportFormat::Csv);

        assert_eq!(generator.format, ReportFormat::Csv);
    }
}
