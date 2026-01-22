import React from 'react';
import { CompressionHistory } from '../../types';
import { useChartVisibility } from '../../hooks';
import SourceSimilarities from '../SourceSimilarities';

interface SourceSimilaritiesContainerProps {
  data: CompressionHistory;
}

const SourceSimilaritiesContainer: React.FC<SourceSimilaritiesContainerProps> = ({ data }) => {
  const { showSourceSimilarities, chartData } = useChartVisibility(data);

  if (!showSourceSimilarities || !chartData.sourceSimilarities) {
    return null;
  }

  return (
    <section className="chart-section">
      <h2 className="section-title">Source Task Similarities</h2>
      <SourceSimilarities
        similarities={chartData.sourceSimilarities}
        taskNames={chartData.taskNames || undefined}
      />
    </section>
  );
};

export default SourceSimilaritiesContainer;
