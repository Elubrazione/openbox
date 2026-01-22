import React from 'react';
import { CompressionHistory } from '../../types';
import { useCompressionPipeline, useChartVisibility } from '../../hooks';
import ParameterImportance from '../ParameterImportance';

interface ParameterImportanceContainerProps {
  data: CompressionHistory;
  
  topK?: number;
}

const ParameterImportanceContainer: React.FC<ParameterImportanceContainerProps> = ({
  data,
  topK = 20,
}) => {
  const { event } = useCompressionPipeline(data);
  const { showParameterImportance, chartData } = useChartVisibility(data);
  if (!showParameterImportance || !chartData.paramImportances || !event) {
    return null;
  }

  return (
    <section className="chart-section">
      <h2 className="section-title">Parameter Importance Analysis</h2>
      <ParameterImportance
        paramNames={event.spaces.original.parameters}
        importances={chartData.paramImportances}
        topK={Math.min(topK, event.spaces.original.parameters.length)}
      />
    </section>
  );
};

export default ParameterImportanceContainer;
