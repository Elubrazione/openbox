import type { ChartPreset, PresetName } from '../types';
import { defaultGrid, spaciousGrid } from './theme';

export const chartPresets: Record<PresetName, ChartPreset> = {
  parameterImportance: {
    name: 'Parameter Importance',
    type: 'horizontalBar',
    context: {
      title: { text: 'Top Parameter Importance' },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: {
        left: '5%',
        right: '15%',
        top: '10%',
        bottom: '5%',
        containLabel: true,
      },
    },
    description: '水平柱状图展示参数重要性排名',
  },
  dimensionReduction: {
    name: 'Dimension Reduction',
    type: 'bar',
    context: {
      title: { text: 'Dimension Reduction Across Steps' },
      grid: spaciousGrid,
    },
    description: '柱状图展示各步骤的维度变化',
  },
  compressionRatio: {
    name: 'Compression Ratio',
    type: 'bar',
    context: {
      title: { text: 'Compression Ratio by Step' },
      grid: spaciousGrid,
    },
    description: '柱状图展示各步骤的压缩比率',
  },
  rangeCompression: {
    name: 'Range Compression',
    type: 'custom',
    context: {
      title: { text: 'Range Compression Details' },
      grid: {
        left: '5%',
        right: '20%',
        top: '60',
        bottom: '30',
        containLabel: true,
      },
    },
    description: '自定义图表展示参数范围压缩详情',
  },
  dimensionEvolution: {
    name: 'Dimension Evolution',
    type: 'line',
    context: {
      title: { text: 'Dimension Evolution Over Iterations' },
      grid: defaultGrid,
    },
    description: '折线图展示迭代过程中的维度变化',
  },
  multiTaskHeatmap: {
    name: 'Multi-Task Heatmap',
    type: 'heatmap',
    context: {
      title: { text: 'Multi-Task Parameter Importance' },
      grid: {
        left: '20%',
        right: '10%',
        top: '15%',
        bottom: '20%',
        containLabel: true,
      },
    },
    description: '热力图展示多任务参数重要性',
  },
};

export function getPreset(name: PresetName): ChartPreset | undefined {
  return chartPresets[name];
}

export function getPresetNames(): PresetName[] {
  return Object.keys(chartPresets) as PresetName[];
}
