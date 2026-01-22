import React from 'react';
import ReactECharts from 'echarts-for-react';
import echarts from '../utils/echarts';
import { useDimensionEvolutionConfig } from '../charts';

interface DimensionEvolutionProps {
  iterations: number[];
  dimensions: number[];
}

const DimensionEvolution: React.FC<DimensionEvolutionProps> = ({
  iterations,
  dimensions,
}) => {
  const { option, dimensions: chartDimensions, isValid, error } = useDimensionEvolutionConfig(
    iterations,
    dimensions
  );

  if (!isValid) {
    return (
      <div style={{ 
        width: '100%', 
        background: '#fff', 
        padding: '40px', 
        borderRadius: '8px',
        textAlign: 'center',
        color: '#999',
      }}>
        {error || 'No data available'}
      </div>
    );
  }

  return (
    <div style={{ 
      width: '100%', 
      background: '#fff', 
      padding: '20px', 
      borderRadius: '8px', 
      marginBottom: '20px',
    }}>
      <ReactECharts 
        echarts={echarts}
        option={option} 
        style={{ height: chartDimensions.height }}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
};

export default DimensionEvolution;
