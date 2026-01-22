import type { EChartsOption } from 'echarts';
import type { ChartType, ChartConfigContext, BarChartData, GridConfig } from '../types';
import { BaseChartStrategy } from './base';
import { spaciousGrid, getCompressionRatioColor } from '../config';

export class BarChartStrategy extends BaseChartStrategy<BarChartData> {
  readonly type: ChartType = 'bar';
  
  validateData(data: unknown): data is BarChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<BarChartData>;
    return Array.isArray(d.categories) && Array.isArray(d.values);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return { ...super.getDefaultContext(), grid: spaciousGrid };
  }
  
  protected buildChartOption(data: BarChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { categories, values, colors, labels = true } = data;
    const palette = context.colors!;
    const barColors = colors || values.map((_, i) => `rgba(64, 158, 255, ${0.4 + (i / values.length) * 0.6})`);
    
    return {
      xAxis: {
        type: 'category', data: categories,
        axisLabel: { rotate: categories.length > 5 ? 45 : 0, fontSize: 10, interval: 0, overflow: 'truncate', width: 80 },
      },
      yAxis: { type: 'value', nameTextStyle: { fontWeight: 'bold' } },
      series: [{
        type: 'bar',
        data: values.map((value, idx) => ({ value, itemStyle: { color: barColors[idx] || palette.primary } })),
        label: labels ? { show: true, position: 'top', fontWeight: 'bold' } : undefined,
        barMaxWidth: 50,
      }],
    };
  }
}

export class HorizontalBarChartStrategy extends BaseChartStrategy<BarChartData> {
  readonly type: ChartType = 'horizontalBar';
  
  validateData(data: unknown): data is BarChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<BarChartData>;
    return Array.isArray(d.categories) && Array.isArray(d.values);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return {
      ...super.getDefaultContext(),
      grid: { left: '5%', right: '15%', top: '10%', bottom: '5%', containLabel: true } as GridConfig,
    };
  }
  
  protected buildChartOption(data: BarChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { categories, values, colors, labels = true } = data;
    const palette = context.colors!;
    const barColor = colors?.[0] || palette.series[0] || '#ff7875';
    
    return {
      xAxis: { type: 'value', nameLocation: 'middle', nameGap: 30, nameTextStyle: { fontWeight: 'bold', fontSize: 12 } },
      yAxis: { type: 'category', data: categories, axisLabel: { fontSize: 11 }, inverse: true },
      series: [{
        type: 'bar', data: values, itemStyle: { color: barColor },
        label: labels ? { show: true, position: 'right', formatter: (params: any) => params.value.toFixed(4), fontSize: 10 } : undefined,
        barMaxWidth: 30,
      }],
    };
  }
}

export class CompressionRatioBarStrategy extends BaseChartStrategy<BarChartData> {
  readonly type: ChartType = 'bar';
  
  validateData(data: unknown): data is BarChartData {
    if (!data || typeof data !== 'object') return false;
    const d = data as Partial<BarChartData>;
    return Array.isArray(d.categories) && Array.isArray(d.values);
  }
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return { ...super.getDefaultContext(), grid: spaciousGrid };
  }
  
  protected buildChartOption(data: BarChartData, context: ChartConfigContext): Partial<EChartsOption> {
    const { categories, values } = data;
    return {
      xAxis: {
        type: 'category', data: categories,
        axisLabel: { rotate: 45, fontSize: 10, interval: 0, overflow: 'truncate', width: 80 },
      },
      yAxis: { type: 'value', name: 'Ratio', nameTextStyle: { fontWeight: 'bold' }, max: 1 },
      series: [{
        type: 'bar',
        data: values.map(ratio => ({ value: ratio, itemStyle: { color: getCompressionRatioColor(ratio) } })),
        label: { show: true, position: 'top', formatter: (params: any) => `${(params.value * 100).toFixed(1)}%`, fontWeight: 'bold' },
        barMaxWidth: 50,
      }],
    };
  }
}
