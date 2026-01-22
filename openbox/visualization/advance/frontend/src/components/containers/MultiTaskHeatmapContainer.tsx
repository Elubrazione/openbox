import React from 'react';
import { CompressionHistory } from '../../types';
import { useCompressionPipeline, useChartVisibility } from '../../hooks';
import MultiTaskHeatmap from '../MultiTaskHeatmap';

interface MultiTaskHeatmapContainerProps {
  data: CompressionHistory;
}

const MultiTaskHeatmapContainer: React.FC<MultiTaskHeatmapContainerProps> = ({ data }) => {
  const { event } = useCompressionPipeline(data);
  const { showMultiTaskHeatmap, chartData } = useChartVisibility(data);

  if (!showMultiTaskHeatmap || !chartData.multiTaskImportances || !event) {
    return null;
  }

  return (
    <section className="chart-section">
      <h2 className="section-title">Multi-Task Parameter Importance</h2>
      <MultiTaskHeatmap
        paramNames={event.spaces.original.parameters}
        importances={chartData.multiTaskImportances}
        tasks={chartData.taskNames || undefined}
      />
    </section>
  );
};

export default MultiTaskHeatmapContainer;
