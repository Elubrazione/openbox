import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import echarts from '../utils/echarts';
import { CompressionHistory } from '../types';
import { useCompressionPipeline } from '../hooks';
import { 
  useDimensionReductionConfig, 
  useCompressionRatioConfig,
  useBarChartConfig,
  chartHeights,
} from '../charts';

interface CompressionSummaryProps {
  data: CompressionHistory;
}

const CompressionSummary: React.FC<CompressionSummaryProps> = ({ data }) => {
  const { event, activeSteps, getCompressionStats } = useCompressionPipeline(data);
  const stats = useMemo(() => getCompressionStats(), [getCompressionStats]);
  const stepNames = useMemo(() => 
    stats ? ['Original', ...activeSteps.map(s => s.name)] : null,
    [stats, activeSteps]
  );

  const compressionStepNames = useMemo(() =>
    activeSteps.map(s => s.name),
    [activeSteps]
  );

  const compressionRatios = useMemo(() => 
    stats ? stats.dimensionFlow.slice(1).map(dim => dim / stats.originalDim) : null,
    [stats]
  );
  const { option: dimReductionOption } = useDimensionReductionConfig(
    stepNames,
    stats?.dimensionFlow ?? null
  );

  const { option: ratioOption } = useCompressionRatioConfig(
    compressionStepNames,
    compressionRatios
  );
  const rangeCompressionData = useMemo(() => {
    const compressionStep = activeSteps.find(s =>
      s.compression_info &&
      s.compression_info.compressed_params &&
      s.compression_info.compressed_params.length > 0
    );

    if (!compressionStep?.compression_info) return null;

    return {
      step: compressionStep,
      nCompressed: compressionStep.compression_info.compressed_params?.length || 0,
      nUnchanged: compressionStep.compression_info.unchanged_params?.length || 0,
    };
  }, [activeSteps]);
  const { option: rangeStatsOption } = useBarChartConfig(
    rangeCompressionData ? {
      categories: [`Step ${rangeCompressionData.step.step_index + 1}\n${rangeCompressionData.step.name}`],
      values: [rangeCompressionData.nCompressed],
    } : null,
    {
      title: { text: 'Range/Quantization Compression Statistics' },
    }
  );
  const getSummaryText = useMemo(() => {
    if (!event || !stats) return '';

    let text = `Compression Summary\n${'='.repeat(40)}\n\n`;
    text += `Original dimensions: ${stats.originalDim}\n`;
    text += `Final sample space: ${event.spaces.sample.n_parameters}\n`;
    text += `Final surrogate space: ${stats.finalDim}\n`;
    text += `Overall compression: ${(stats.ratio * 100).toFixed(1)}%\n\n`;
    text += `Active Steps: ${stats.stepCount}\n`;

    activeSteps.forEach((step, i) => {
      const inputDim = stats.dimensionFlow[i];
      const outputDim = stats.dimensionFlow[i + 1];
      const dimRatio = outputDim / inputDim;

      text += `${i + 1}. ${step.name}\n`;
      text += `   ${inputDim} → ${outputDim} (${(dimRatio * 100).toFixed(1)}%)\n`;

      if (step.compression_info?.avg_compression_ratio) {
        text += `   Effective: ${(step.compression_info.avg_compression_ratio * 100).toFixed(1)}%\n`;
      }
    });

    return text;
  }, [event, stats, activeSteps]);
  if (!event || !stats) {
    return (
      <div style={{ width: '100%', background: '#fff', padding: '40px', borderRadius: '8px', textAlign: 'center' }}>
        <p style={{ color: '#999' }}>No compression data available</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', background: '#fff', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '20px', fontSize: '20px', fontWeight: 'bold' }}>
        Compression Summary
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>
          <ReactECharts 
            echarts={echarts}
            option={dimReductionOption} 
            style={{ height: chartHeights.medium }}
            notMerge={true}
            lazyUpdate={true}
          />
        </div>
        <div>
          <ReactECharts 
            echarts={echarts}
            option={ratioOption} 
            style={{ height: chartHeights.medium }}
            notMerge={true}
            lazyUpdate={true}
          />
        </div>
        <div>
          {rangeCompressionData ? (
            <ReactECharts 
              echarts={echarts}
              option={rangeStatsOption} 
              style={{ height: chartHeights.medium }}
              notMerge={true}
              lazyUpdate={true}
            />
          ) : (
            <div style={{ 
              height: chartHeights.medium, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              background: '#fafafa',
              borderRadius: '4px',
            }}>
              <p style={{ color: '#999' }}>No range/quantization compression data</p>
            </div>
          )}
        </div>
        <div
          style={{
            padding: '20px',
            background: '#fffbf0',
            borderRadius: '8px',
            border: '1px solid #ffe58f',
            fontFamily: 'monospace',
            fontSize: '11px',
            whiteSpace: 'pre-wrap',
            overflowY: 'auto',
            height: chartHeights.medium,
          }}
        >
          {getSummaryText}
        </div>
      </div>
    </div>
  );
};

export default CompressionSummary;
