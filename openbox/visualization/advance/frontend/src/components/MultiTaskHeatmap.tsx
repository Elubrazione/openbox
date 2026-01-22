import React from 'react';
import ReactECharts from 'echarts-for-react';
import echarts from '../utils/echarts';
import { useMultiTaskHeatmapConfig } from '../charts';

interface MultiTaskHeatmapProps {
  paramNames: string[];
  importances: number[][];
  tasks?: string[];
}

const MultiTaskHeatmap: React.FC<MultiTaskHeatmapProps> = ({
  paramNames,
  importances,
  tasks,
}) => {
  const taskNames = tasks || importances.map((_, i) => `Task ${i + 1}`);
  const { option, dimensions, isValid, error } = useMultiTaskHeatmapConfig(
    paramNames,
    taskNames,
    importances
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
        style={{ height: dimensions.height }}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
};

export default MultiTaskHeatmap;
