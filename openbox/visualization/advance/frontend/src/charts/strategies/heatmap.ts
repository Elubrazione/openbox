import type { EChartsOption } from 'echarts';
import type { ChartType, ChartConfigContext, HeatmapChartData } from '../types';
import { BaseChartStrategy } from './base';

export class HeatmapChartStrategy extends BaseChartStrategy<HeatmapChartData> {
  readonly type: ChartType = 'heatmap';
  
  validateData(data: unknown): data is HeatmapChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<HeatmapChartData>;
    return Array.isArray(d.xCategories) && Array.isArray(d.yCategories) && Array.isArray(d.values);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return {
      ...super.getDefaultContext(),
      grid: { left: '15%', right: '15%', top: '10%', bottom: '15%', containLabel: true },
    };
  }
  
  protected buildChartOption(data: HeatmapChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { xCategories, yCategories, values, min, max } = data;
    
    const heatmapData: [number, number, number][] = [];
    let dataMin = min ?? Infinity;
    let dataMax = max ?? -Infinity;
    
    values.forEach((row, yIdx) => {
      row.forEach((value, xIdx) => {
        heatmapData.push([xIdx, yIdx, value]);
        if (min === undefined) dataMin = Math.min(dataMin, value);
        if (max === undefined) dataMax = Math.max(dataMax, value);
      });
    });
    
    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const [x, y, value] = params.data;
          return `${yCategories[y]} / ${xCategories[x]}<br/>Value: ${value.toFixed(4)}`;
        },
      },
      xAxis: { type: 'category', data: xCategories, splitArea: { show: true }, axisLabel: { rotate: 45, fontSize: 10, interval: 0 } },
      yAxis: { type: 'category', data: yCategories, splitArea: { show: true }, axisLabel: { fontSize: 10 } },
      visualMap: {
        min: dataMin, max: dataMax, calculable: true, orient: 'horizontal', left: 'center', bottom: '5%',
        inRange: { color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'] },
      },
      series: [{
        type: 'heatmap', data: heatmapData,
        label: { show: heatmapData.length <= 100, formatter: (params: any) => params.data[2].toFixed(2), fontSize: 9 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
      }],
    };
  }
}

export class MultiTaskHeatmapStrategy extends BaseChartStrategy<HeatmapChartData> {
  readonly type: ChartType = 'heatmap';
  
  validateData(data: unknown): data is HeatmapChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<HeatmapChartData>;
    return Array.isArray(d.xCategories) && Array.isArray(d.yCategories) && Array.isArray(d.values);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return {
      ...super.getDefaultContext(),
      title: { text: 'Multi-Task Parameter Importance' },
      grid: { left: '20%', right: '10%', top: '15%', bottom: '20%', containLabel: true },
    };
  }
  
  protected buildChartOption(data: HeatmapChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { xCategories, yCategories, values } = data;
    const paramNames = xCategories.map(name => name.split('.').pop() || name);
    
    const heatmapData: [number, number, number][] = [];
    let dataMin = Infinity, dataMax = -Infinity;
    
    values.forEach((row, yIdx) => {
      row.forEach((value, xIdx) => {
        heatmapData.push([xIdx, yIdx, value]);
        dataMin = Math.min(dataMin, value);
        dataMax = Math.max(dataMax, value);
      });
    });
    
    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const [x, y, value] = params.data;
          return `Task: ${yCategories[y]}<br/>Param: ${xCategories[x]}<br/>Importance: ${value.toFixed(4)}`;
        },
      },
      xAxis: {
        type: 'category', data: paramNames, splitArea: { show: true },
        axisLabel: { rotate: 60, fontSize: 9, interval: 0 },
        name: 'Parameters', nameLocation: 'middle', nameGap: 50,
      },
      yAxis: {
        type: 'category', data: yCategories, splitArea: { show: true },
        axisLabel: { fontSize: 10 }, name: 'Tasks', nameLocation: 'middle', nameGap: 80,
      },
      visualMap: {
        min: dataMin, max: dataMax, calculable: true, orient: 'vertical', right: '2%', top: 'center',
        text: ['High', 'Low'],
        inRange: { color: ['#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695'] },
      },
      series: [{
        type: 'heatmap', data: heatmapData,
        label: { show: heatmapData.length <= 50, formatter: (params: any) => params.data[2].toFixed(2), fontSize: 8 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } },
      }],
    };
  }
}
