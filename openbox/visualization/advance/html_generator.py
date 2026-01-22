"""
OpenBox Static HTML Generator

Generates standalone HTML files for basic visualization.
No server required - can be opened directly in a browser.
"""

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from openbox import logger


def generate_static_html(
    data_dir: str,
    output_path: Optional[str] = None,
    open_browser: bool = True
) -> str:
    """
    Generate a standalone HTML visualization file.
    
    This creates a single HTML file that can be opened in any browser
    without requiring a server.
    
    Args:
        data_dir: Directory containing compression_history.json
        output_path: Where to save the HTML file (default: data_dir/visualization.html)
        open_browser: Whether to open the file in browser
    
    Returns:
        Path to the generated HTML file
    
    Example:
        >>> html_path = generate_static_html('./results/compression')
        >>> print(f"Visualization saved to {html_path}")
    """
    data_dir = Path(data_dir)
    
    # Load compression history
    history_file = data_dir / 'compression_history.json'
    if not history_file.exists():
        raise FileNotFoundError(f"Compression history not found at {history_file}")
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
    
    # Generate HTML
    html_content = _generate_html_content(history_data)
    
    # Determine output path
    if output_path is None:
        output_path = data_dir / 'visualization.html'
    else:
        output_path = Path(output_path)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Visualization saved to {output_path}")
    
    # Open in browser if requested
    if open_browser:
        try:
            webbrowser.open(f'file://{output_path.absolute()}')
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            logger.info(f"Please open the file manually: {output_path.absolute()}")
    
    return str(output_path.absolute())


def _generate_html_content(data: Dict[str, Any]) -> str:
    """Generate the complete HTML content with embedded data and ECharts."""
    
    # Serialize data to JSON
    data_json = json.dumps(data, indent=2)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Use regular string formatting instead of f-string to avoid issues with JS template literals
    html_template = _get_html_template()
    
    # Replace placeholders
    html = html_template.replace('__DATA_JSON__', data_json)
    html = html.replace('__TIMESTAMP__', timestamp)
    
    return html


def _get_html_template() -> str:
    """Return the HTML template with placeholders."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenBox Compression Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #67c23a;
            --warning-color: #e6a23c;
            --danger-color: #f56c6c;
            --bg-color: #f5f7fa;
            --card-bg: #ffffff;
            --text-color: #303133;
            --text-secondary: #606266;
            --border-color: #dcdfe6;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .app-header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        
        .app-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }
        
        .app-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .chart-section {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid var(--border-color);
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--primary-color);
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        
        .chart-item {
            min-height: 350px;
        }
        
        .summary-text {
            padding: 20px;
            background: #fffbf0;
            border-radius: 8px;
            border: 1px solid #ffe58f;
            font-family: 'SF Mono', SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 12px;
            white-space: pre-wrap;
            overflow-y: auto;
            max-height: 350px;
        }
        
        .app-footer {
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .timestamp {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        @media (max-width: 900px) {
            .chart-grid {
                grid-template-columns: 1fr;
            }
            .app-header h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="app-header">
            <h1>📊 OpenBox Compression Visualization</h1>
            <p>Interactive visualization of compression history and parameter space analysis</p>
            <p class="timestamp">Generated: __TIMESTAMP__</p>
        </header>
        
        <main>
            <!-- Compression Summary -->
            <section class="chart-section">
                <h2 class="section-title">Compression Summary</h2>
                <div class="chart-grid">
                    <div id="dimension-chart" class="chart-item"></div>
                    <div id="ratio-chart" class="chart-item"></div>
                    <div id="stats-chart" class="chart-item"></div>
                    <div id="summary-text" class="summary-text"></div>
                </div>
            </section>
            
            <!-- Range Compression -->
            <section class="chart-section" id="range-section" style="display:none;">
                <h2 class="section-title">Range Compression Details</h2>
                <div id="range-chart" style="min-height: 400px;"></div>
            </section>
            
            <!-- Parameter Importance -->
            <section class="chart-section" id="importance-section" style="display:none;">
                <h2 class="section-title">Parameter Importance Analysis</h2>
                <div id="importance-chart" style="min-height: 400px;"></div>
            </section>
            
            <!-- Multi-Task Heatmap -->
            <section class="chart-section" id="heatmap-section" style="display:none;">
                <h2 class="section-title">Multi-Task Parameter Importance</h2>
                <div id="heatmap-chart" style="min-height: 500px;"></div>
            </section>
            
            <!-- Source Similarities -->
            <section class="chart-section" id="similarity-section" style="display:none;">
                <h2 class="section-title">Source Task Similarities</h2>
                <div id="similarity-chart" style="min-height: 350px;"></div>
            </section>
        </main>
        
        <footer class="app-footer">
            <p>OpenBox Visualization | Powered by ECharts</p>
        </footer>
    </div>
    
    <script>
        // Embedded compression data
        const COMPRESSION_DATA = __DATA_JSON__;
        
        // Initialize visualization
        document.addEventListener('DOMContentLoaded', function() {
            if (!COMPRESSION_DATA || !COMPRESSION_DATA.history || COMPRESSION_DATA.history.length === 0) {
                document.querySelector('main').innerHTML = '<div class="chart-section"><p>No compression data available.</p></div>';
                return;
            }
            
            const event = COMPRESSION_DATA.history[0];
            const pipeline = event.pipeline;
            
            // Filter active steps
            const activeSteps = pipeline.steps.filter((step, index) => {
                const inputDim = index === 0 
                    ? event.spaces.original.n_parameters 
                    : pipeline.steps[index - 1].output_space_params;
                const isNoneStep = step.name.toLowerCase().includes('none');
                const hasCompressionInfo = step.compression_info && 
                    (step.compression_info.compressed_params?.length > 0);
                const isProjectionStep = step.name.toLowerCase().includes('projection') ||
                    step.type.toLowerCase().includes('projection');
                const isUselessProjection = isProjectionStep && inputDim === step.output_space_params && !hasCompressionInfo;
                return !isNoneStep && !isUselessProjection && step.output_space_params > 0;
            });
            
            // Render charts
            renderDimensionChart(event, activeSteps);
            renderRatioChart(event, activeSteps);
            renderStatsChart(activeSteps);
            renderSummaryText(event, activeSteps);
            
            // Render range compression details
            const rangeStep = activeSteps.find(s => 
                s.compression_info && s.compression_info.compressed_params?.length > 0
            );
            if (rangeStep) {
                document.getElementById('range-section').style.display = 'block';
                renderRangeChart(rangeStep);
            }
            
            // Render parameter importance if available
            const hasImportanceStep = activeSteps.some(s => 
                s.type.includes('SHAP') || s.type.includes('Correlation') || s.type.includes('Adaptive')
            );
            if (hasImportanceStep) {
                document.getElementById('importance-section').style.display = 'block';
                renderImportanceChart(event);
            }
            
            // Render multi-task heatmap if available
            if (event.performance_metrics?.multi_task_importances) {
                document.getElementById('heatmap-section').style.display = 'block';
                renderHeatmapChart(event);
            }
            
            // Render source similarities if available
            if (event.performance_metrics?.source_similarities) {
                document.getElementById('similarity-section').style.display = 'block';
                renderSimilarityChart(event);
            }
        });
        
        function renderDimensionChart(event, activeSteps) {
            const chart = echarts.init(document.getElementById('dimension-chart'));
            const stepNames = ['Original', ...activeSteps.map(s => s.name)];
            const dimensions = [
                event.spaces.original.n_parameters,
                ...activeSteps.map(s => s.output_space_params)
            ];
            
            chart.setOption({
                title: {
                    text: 'Dimension Reduction Across Steps',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                xAxis: {
                    type: 'category',
                    data: stepNames,
                    axisLabel: { rotate: 45, fontSize: 10, interval: 0 }
                },
                yAxis: {
                    type: 'value',
                    name: 'Parameters',
                    nameTextStyle: { fontWeight: 'bold' }
                },
                grid: { left: '12%', right: '8%', bottom: '25%', top: '15%' },
                series: [{
                    type: 'bar',
                    data: dimensions.map((dim, idx) => ({
                        value: dim,
                        itemStyle: { color: `rgba(102, 126, 234, ${0.4 + idx * 0.15})` }
                    })),
                    label: { show: true, position: 'top', fontWeight: 'bold' },
                    barMaxWidth: 50
                }]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderRatioChart(event, activeSteps) {
            const chart = echarts.init(document.getElementById('ratio-chart'));
            const originalDim = event.spaces.original.n_parameters;
            const ratios = activeSteps.map(s => ({
                name: s.name,
                ratio: s.output_space_params / originalDim
            }));
            
            chart.setOption({
                title: {
                    text: 'Compression Ratio by Step',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: params => `${params[0].name}<br/>Ratio: ${(params[0].value * 100).toFixed(1)}%`
                },
                xAxis: {
                    type: 'category',
                    data: ratios.map(r => r.name),
                    axisLabel: { rotate: 45, fontSize: 10, interval: 0 }
                },
                yAxis: {
                    type: 'value',
                    name: 'Ratio',
                    max: 1
                },
                grid: { left: '12%', right: '8%', bottom: '25%', top: '15%' },
                series: [{
                    type: 'bar',
                    data: ratios.map(r => ({
                        value: r.ratio,
                        itemStyle: {
                            color: r.ratio > 0.7 ? '#f56c6c' : r.ratio > 0.4 ? '#e6a23c' : '#67c23a'
                        }
                    })),
                    label: {
                        show: true,
                        position: 'top',
                        formatter: p => `${(p.value * 100).toFixed(1)}%`,
                        fontWeight: 'bold'
                    },
                    barMaxWidth: 50
                }]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderStatsChart(activeSteps) {
            const chart = echarts.init(document.getElementById('stats-chart'));
            const compressionStep = activeSteps.find(s => 
                s.compression_info?.compressed_params?.length > 0
            );
            
            if (!compressionStep) {
                chart.setOption({
                    title: { text: 'Range Compression Statistics', left: 'center' },
                    graphic: {
                        type: 'text',
                        left: 'center',
                        top: 'middle',
                        style: { text: 'No range compression data', fontSize: 14, fill: '#999' }
                    }
                });
                return;
            }
            
            const nCompressed = compressionStep.compression_info.compressed_params.length;
            const nUnchanged = compressionStep.compression_info.unchanged_params?.length || 0;
            
            chart.setOption({
                title: {
                    text: 'Range Compression Statistics',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: { trigger: 'axis' },
                legend: { data: ['Compressed', 'Unchanged'], bottom: 0 },
                xAxis: {
                    type: 'category',
                    data: [`Step ${compressionStep.step_index + 1}\n${compressionStep.name}`]
                },
                yAxis: { type: 'value', name: 'Parameters' },
                grid: { left: '12%', right: '8%', bottom: '15%', top: '15%' },
                series: [
                    {
                        name: 'Compressed',
                        type: 'bar',
                        stack: 'total',
                        data: [nCompressed],
                        itemStyle: { color: '#ff7875' },
                        label: { show: true, position: 'inside' }
                    },
                    {
                        name: 'Unchanged',
                        type: 'bar',
                        stack: 'total',
                        data: [nUnchanged],
                        itemStyle: { color: '#91d5ff' },
                        label: { show: true, position: 'inside' }
                    }
                ]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderSummaryText(event, activeSteps) {
            const originalDim = event.spaces.original.n_parameters;
            const dimensions = [originalDim, ...activeSteps.map(s => s.output_space_params)];
            
            let text = 'Compression Summary\\n' + '='.repeat(40) + '\\n\\n';
            text += 'Original dimensions: ' + originalDim + '\\n';
            text += 'Final sample space: ' + event.spaces.sample.n_parameters + '\\n';
            text += 'Final surrogate space: ' + event.spaces.surrogate.n_parameters + '\\n';
            text += 'Overall compression: ' + (event.spaces.surrogate.n_parameters / originalDim * 100).toFixed(1) + '%\\n\\n';
            text += 'Active Steps: ' + activeSteps.length + '\\n';
            
            activeSteps.forEach((step, i) => {
                const inputDim = dimensions[i];
                const outputDim = dimensions[i + 1];
                const ratio = outputDim / inputDim;
                
                text += '\\n' + (i + 1) + '. ' + step.name + '\\n';
                text += '   ' + inputDim + ' → ' + outputDim + ' (' + (ratio * 100).toFixed(1) + '%)\\n';
                
                if (step.compression_info && step.compression_info.avg_compression_ratio) {
                    text += '   Effective: ' + (step.compression_info.avg_compression_ratio * 100).toFixed(1) + '%\\n';
                }
            });
            
            document.getElementById('summary-text').textContent = text;
        }
        
        function renderRangeChart(step) {
            const chart = echarts.init(document.getElementById('range-chart'));
            const params = step.compression_info.compressed_params.slice(0, 30);
            
            const chartData = params.map((p, idx) => {
                const origMin = p.original_range?.[0] ?? 0;
                const origMax = p.original_range?.[1] ?? 1;
                const compMin = p.compressed_range?.[0] ?? origMin;
                const compMax = p.compressed_range?.[1] ?? origMax;
                const isQuant = p.original_num_values !== undefined;
                
                let normStart = 0, normEnd = 1;
                if (!isQuant && origMax - origMin > 0) {
                    normStart = (compMin - origMin) / (origMax - origMin);
                    normEnd = (compMax - origMin) / (origMax - origMin);
                }
                
                return {
                    name: p.name,
                    ratio: p.compression_ratio,
                    normStart,
                    normEnd,
                    compMin,
                    compMax,
                    isQuant
                };
            });
            
            chart.setOption({
                title: {
                    text: `${step.name}: Range Compression Details`,
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 16 }
                },
                tooltip: {
                    trigger: 'item',
                    formatter: p => {
                        const d = chartData[p.dataIndex];
                        return `${d.name}<br/>Ratio: ${(d.ratio * 100).toFixed(1)}%<br/>Range: [${d.compMin.toFixed(2)}, ${d.compMax.toFixed(2)}]`;
                    }
                },
                grid: { left: '15%', right: '15%', top: 60, bottom: 30 },
                xAxis: {
                    type: 'value',
                    name: 'Normalized Range',
                    min: -0.1,
                    max: 1.3
                },
                yAxis: {
                    type: 'category',
                    data: chartData.map(d => d.name),
                    axisLabel: { fontSize: 10 }
                },
                series: [
                    {
                        name: 'Original',
                        type: 'bar',
                        data: chartData.map(() => 1),
                        itemStyle: { color: 'rgba(150,150,150,0.3)' },
                        barWidth: '40%',
                        z: 1
                    },
                    {
                        name: 'Compressed',
                        type: 'custom',
                        renderItem: (params, api) => {
                            const d = chartData[params.dataIndex];
                            const start = api.coord([d.normStart, params.dataIndex]);
                            const end = api.coord([d.normEnd, params.dataIndex]);
                            const height = api.size([0, 1])[1] * 0.35;
                            
                            let color = '#67c23a';
                            if (d.ratio > 0.9) color = '#f56c6c';
                            else if (d.ratio > 0.7) color = '#e6a23c';
                            else if (d.ratio > 0.5) color = '#f0a020';
                            
                            return {
                                type: 'rect',
                                shape: { x: start[0], y: start[1] - height/2, width: end[0] - start[0], height },
                                style: { fill: color, opacity: 0.8 }
                            };
                        },
                        data: chartData.map((d, i) => [d.normStart, i, d.normEnd - d.normStart]),
                        z: 2
                    },
                    {
                        name: 'Labels',
                        type: 'custom',
                        renderItem: (params, api) => {
                            const d = chartData[params.dataIndex];
                            const pos = api.coord([1.05, params.dataIndex]);
                            return {
                                type: 'text',
                                x: pos[0],
                                y: pos[1],
                                style: {
                                    text: `${(d.ratio * 100).toFixed(1)}%`,
                                    fontSize: 10,
                                    fill: '#666'
                                }
                            };
                        },
                        data: chartData.map((d, i) => [0, i]),
                        z: 3
                    }
                ]
            });
            
            document.getElementById('range-chart').style.height = `${Math.max(400, params.length * 25)}px`;
            chart.resize();
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderImportanceChart(event) {
            const chart = echarts.init(document.getElementById('importance-chart'));
            const params = event.spaces.original.parameters;
            
            // Generate mock importance data (in real use, this would come from the actual data)
            const importances = params.map(() => 0.1 + Math.random() * 0.9);
            const sortedData = params.map((name, i) => ({ name, importance: importances[i] }))
                .sort((a, b) => b.importance - a.importance)
                .slice(0, 20);
            
            chart.setOption({
                title: {
                    text: 'Top-20 Parameter Importance',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                grid: { left: '20%', right: '10%', top: 60, bottom: 30 },
                xAxis: {
                    type: 'value',
                    name: 'Importance Score'
                },
                yAxis: {
                    type: 'category',
                    data: sortedData.map(d => d.name).reverse(),
                    axisLabel: { fontSize: 10 }
                },
                series: [{
                    type: 'bar',
                    data: sortedData.map(d => d.importance).reverse(),
                    itemStyle: { color: '#ff7875' },
                    label: {
                        show: true,
                        position: 'right',
                        formatter: p => p.value.toFixed(4)
                    }
                }]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderHeatmapChart(event) {
            const chart = echarts.init(document.getElementById('heatmap-chart'));
            const importances = event.performance_metrics.multi_task_importances;
            const taskNames = event.performance_metrics.task_names || 
                importances.map((_, i) => `Task ${i + 1}`);
            const params = event.spaces.original.parameters.slice(0, 30);
            
            // Normalize and prepare heatmap data
            const maxVal = Math.max(...importances.flat());
            const heatmapData = [];
            
            importances.forEach((taskImportances, taskIdx) => {
                taskImportances.slice(0, 30).forEach((val, paramIdx) => {
                    heatmapData.push([paramIdx, taskIdx, (val / maxVal).toFixed(3)]);
                });
            });
            
            chart.setOption({
                title: {
                    text: 'Multi-Task Parameter Importance Heatmap',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: {
                    position: 'top',
                    formatter: p => `${params[p.data[0]]}<br/>${taskNames[p.data[1]]}<br/>Importance: ${p.data[2]}`
                },
                grid: { left: '15%', right: '15%', top: 60, bottom: '25%' },
                xAxis: {
                    type: 'category',
                    data: params,
                    axisLabel: { rotate: 45, fontSize: 9, interval: 0 }
                },
                yAxis: {
                    type: 'category',
                    data: taskNames
                },
                visualMap: {
                    min: 0,
                    max: 1,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: 0,
                    inRange: {
                        color: ['#67c23a', '#f0a020', '#f56c6c']
                    }
                },
                series: [{
                    type: 'heatmap',
                    data: heatmapData,
                    label: { show: false },
                    emphasis: {
                        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' }
                    }
                }]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
        
        function renderSimilarityChart(event) {
            const chart = echarts.init(document.getElementById('similarity-chart'));
            const similarities = event.performance_metrics.source_similarities;
            const taskNames = Object.keys(similarities);
            const values = Object.values(similarities);
            
            chart.setOption({
                title: {
                    text: 'Source Task Similarity to Target Task',
                    left: 'center',
                    textStyle: { fontWeight: 'bold', fontSize: 14 }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: p => `${p[0].name}<br/>Similarity: ${p[0].value.toFixed(3)}`
                },
                xAxis: {
                    type: 'category',
                    data: taskNames,
                    axisLabel: { rotate: 45 }
                },
                yAxis: {
                    type: 'value',
                    name: 'Similarity Score',
                    max: Math.max(...values) * 1.1
                },
                grid: { left: '10%', right: '10%', bottom: '20%', top: '15%' },
                series: [{
                    type: 'bar',
                    data: values.map(v => ({
                        value: v,
                        itemStyle: {
                            color: v > 0.7 ? '#67c23a' : v > 0.4 ? '#e6a23c' : '#f56c6c'
                        }
                    })),
                    label: {
                        show: true,
                        position: 'top',
                        formatter: p => p.value.toFixed(3),
                        fontWeight: 'bold'
                    },
                    barMaxWidth: 50
                }]
            });
            
            window.addEventListener('resize', () => chart.resize());
        }
    </script>
</body>
</html>
'''
