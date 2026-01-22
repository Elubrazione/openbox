import React from 'react';
import { CompressionHistory } from '../../types';
import { useChartVisibility } from '../../hooks';
import DimensionEvolution from '../DimensionEvolution';

interface DimensionEvolutionContainerProps {
  data: CompressionHistory;
}

const DimensionEvolutionContainer: React.FC<DimensionEvolutionContainerProps> = ({ data }) => {
  const { showDimensionEvolution, chartData } = useChartVisibility(data);

  if (!showDimensionEvolution || !chartData.iterations || !chartData.dimensions) {
    return null;
  }

  return (
    <section className="chart-section">
      <h2 className="section-title">Dimension Evolution Over Iterations</h2>
      <DimensionEvolution
        iterations={chartData.iterations}
        dimensions={chartData.dimensions}
      />
    </section>
  );
};

export default DimensionEvolutionContainer;
