import type { EChartsOption } from 'echarts';
import type { ChartType, ChartConfigContext, LineChartData } from '../types';
import { BaseChartStrategy } from './base';

export class LineChartStrategy extends BaseChartStrategy<LineChartData> {
  readonly type: ChartType = 'line';
  
  validateData(data: unknown): data is LineChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<LineChartData>;
    return Array.isArray(d.xData) && Array.isArray(d.yData);
  }
  
  protected buildChartOption(data: LineChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { xData, yData, smooth = true, areaStyle = false } = data;
    const palette = context.colors!;
    
    return {
      xAxis: { type: 'category', data: xData, boundaryGap: false },
      yAxis: { type: 'value', nameTextStyle: { fontWeight: 'bold' } },
      series: [{
        type: 'line', data: yData, smooth, symbol: 'circle', symbolSize: 8,
        itemStyle: { color: palette.primary },
        lineStyle: { width: 2, color: palette.primary },
        areaStyle: areaStyle ? {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: palette.primary + '80' },
              { offset: 1, color: palette.primary + '10' },
            ],
          },
        } : undefined,
      }],
    };
  }
}

export class DimensionEvolutionStrategy extends BaseChartStrategy<LineChartData> {
  readonly type: ChartType = 'line';
  
  validateData(data: unknown): data is LineChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<LineChartData>;
    return Array.isArray(d.xData) && Array.isArray(d.yData);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return {
      ...super.getDefaultContext(),
      title: { text: 'Dimension Evolution Over Iterations' },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const p = params[0];
          return `Iteration ${p.name}<br/>Dimensions: ${p.value}`;
        },
      },
    };
  }
  
  protected buildChartOption(data: LineChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { xData, yData } = data;
    const palette = context.colors!;
    
    return {
      xAxis: {
        type: 'category', data: xData, name: 'Iteration',
        nameLocation: 'middle', nameGap: 30, nameTextStyle: { fontWeight: 'bold' },
      },
      yAxis: { type: 'value', name: 'Dimensions', nameTextStyle: { fontWeight: 'bold' }, minInterval: 1 },
      series: [{
        type: 'line', data: yData, smooth: true, symbol: 'circle', symbolSize: 10,
        itemStyle: { color: palette.primary },
        lineStyle: { width: 3, color: palette.primary },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: palette.primary + '60' },
              { offset: 1, color: palette.primary + '05' },
            ],
          },
        },
        label: { show: true, position: 'top', fontWeight: 'bold', fontSize: 12 },
      }],
    };
  }
}
