import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import echarts from '../utils/echarts';

interface SourceSimilaritiesProps {
  similarities: Record<string, number>; // task_id -> similarity
  taskNames?: string[];
}

const SourceSimilarities: React.FC<SourceSimilaritiesProps> = ({ similarities, taskNames }) => {
  const similarityData = useMemo(() => {
    const entries = Object.entries(similarities || {});

    return entries
      .map(([key, rawValue], index) => {
        const numericValue = Number(rawValue);
        if (!Number.isFinite(numericValue)) {
          return null;
        }

        const numericIndex = Number(key);
        const hasNumericIndex = !Number.isNaN(numericIndex);
        const labelFromTasks =
          hasNumericIndex && taskNames && taskNames[numericIndex] ? taskNames[numericIndex] : undefined;

        return {
          key,
          value: numericValue,
          label: labelFromTasks || key,
          sortKey: hasNumericIndex ? numericIndex : entries.length + index,
        };
      })
      .filter((item): item is { key: string; value: number; label: string; sortKey: number } => item !== null)
      .sort((a, b) => a.sortKey - b.sortKey);
  }, [similarities, taskNames]);

  const labels = similarityData.map(item => item.label);
  const simValues = similarityData.map(item => item.value);
  const maxValue = simValues.length ? Math.max(...simValues) : 1;
  const yAxisMax = maxValue === 0 ? 1 : maxValue * 1.1;

  const getOption = () => {
    return {
      title: {
        text: 'Source Task Similarity to Target Task',
        left: 'center',
        textStyle: { fontWeight: 'bold', fontSize: 16 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const data = params[0];
          return `${data.name}<br/>Similarity: ${data.value.toFixed(3)}`;
        },
      },
      grid: {
        left: '10%',
        right: '10%',
        top: '15%',
        bottom: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          rotate: 45,
          fontSize: 11,
          interval: 0,
        },
        name: 'Source Tasks',
        nameLocation: 'middle',
        nameGap: 60,
        nameTextStyle: { fontWeight: 'bold', fontSize: 12 },
      },
      yAxis: {
        type: 'value',
        name: 'Similarity Score',
        nameTextStyle: { fontWeight: 'bold', fontSize: 12 },
        max: yAxisMax,
      },
      series: [
        {
          type: 'bar',
          data: simValues.map(val => ({
            value: val,
            itemStyle: {
              color:
                val > 0.7
                  ? '#52c41a'
                  : val > 0.4
                    ? '#faad14'
                    : '#ff4d4f',
            },
          })),
          label: {
            show: true,
            position: 'top',
            formatter: (params: any) => Number(params.value).toFixed(3),
            fontWeight: 'bold',
            fontSize: 11,
          },
          barMaxWidth: 60,
        },
      ],
    };
  };

  return (
    <div style={{ width: '100%', background: '#fff', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
      <ReactECharts echarts={echarts} option={getOption()} style={{ height: '450px' }} />
    </div>
  );
};

export default SourceSimilarities;
