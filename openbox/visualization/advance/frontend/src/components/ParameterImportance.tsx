import React from 'react';
import ReactECharts from 'echarts-for-react';
import { useParameterImportanceConfig } from '../charts';

interface ParameterImportanceProps {
  paramNames: string[];
  importances: number[];
  topK?: number;
}

const ParameterImportance: React.FC<ParameterImportanceProps> = ({
  paramNames,
  importances,
  topK = 20,
}) => {
  const { option, dimensions, isValid, error } = useParameterImportanceConfig(
    paramNames,
    importances,
    topK
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
        option={option}
        style={{ height: dimensions.height }}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
};

export default ParameterImportance;
