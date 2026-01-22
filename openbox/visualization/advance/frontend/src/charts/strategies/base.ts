import type { EChartsOption } from 'echarts';
import type { 
  IChartStrategy, 
  ChartType, 
  ChartData, 
  ChartConfigContext,
  TitleConfig,
  TooltipConfig,
  GridConfig,
} from '../types';
import { 
  defaultGrid, 
  defaultTitleStyle, 
  defaultTooltip, 
  defaultAnimation,
  noAnimation,
  getPalette,
} from '../config';

export abstract class BaseChartStrategy<T extends ChartData = ChartData> 
  implements IChartStrategy<T> {
  
  abstract readonly type: ChartType;
  
  createOption(data: T, context: ChartConfigContext): EChartsOption {
    const mergedContext = this.mergeContext(context);
    const baseOption = this.buildBaseOption(mergedContext);
    const chartOption = this.buildChartOption(data, mergedContext);
    return this.deepMerge(baseOption, chartOption);
  }
  
  abstract validateData(data: unknown): data is T;
  protected abstract buildChartOption(data: T, context: ChartConfigContext): Partial<EChartsOption>;
  
  getDefaultContext(): Partial<ChartConfigContext> {
    return { theme: 'light', animation: true, grid: defaultGrid };
  }
  
  protected mergeContext(context: ChartConfigContext): ChartConfigContext {
    const defaults = this.getDefaultContext();
    const palette = getPalette(context.theme || 'light');
    return { ...defaults, ...context, colors: context.colors || palette } as ChartConfigContext;
  }
  
  protected buildBaseOption(context: ChartConfigContext): Partial<EChartsOption> {
    const option: Partial<EChartsOption> = {};
    if (context.title) option.title = this.buildTitleConfig(context.title);
    option.tooltip = this.buildTooltipConfig(context.tooltip);
    option.grid = this.buildGridConfig(context.grid);
    if (context.legend) option.legend = this.buildLegendConfig(context.legend);
    Object.assign(option, context.animation ? defaultAnimation : noAnimation);
    if (context.colors) option.color = context.colors.series;
    return option;
  }
  
  protected buildTitleConfig(config: TitleConfig): EChartsOption['title'] {
    return {
      text: config.text, subtext: config.subtext,
      left: config.left ?? 'center', top: config.top,
      textStyle: { ...defaultTitleStyle, ...config.textStyle } as EChartsOption['title'] extends { textStyle?: infer S } ? S : never,
    };
  }
  
  protected buildTooltipConfig(config?: TooltipConfig): EChartsOption['tooltip'] {
    return { ...defaultTooltip, ...config };
  }
  
  protected buildGridConfig(config?: GridConfig): EChartsOption['grid'] {
    return { ...defaultGrid, ...config };
  }
  
  protected buildLegendConfig(config: NonNullable<ChartConfigContext['legend']>) {
    const positionMap: Record<string, object> = {
      top: { top: '5%', left: 'center' },
      bottom: { bottom: '5%', left: 'center' },
      left: { left: '5%', top: 'middle' },
      right: { right: '5%', top: 'middle' },
    };
    return {
      show: config.show ?? true, data: config.data,
      orient: config.orient ?? 'horizontal',
      ...positionMap[config.position || 'top'],
    };
  }
  
  protected deepMerge<A extends object, B extends object>(target: A, source: B): A & B {
    const output = { ...target } as A & B;
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        const sourceValue = source[key];
        const targetValue = (target as any)[key];
        if (this.isObject(sourceValue) && this.isObject(targetValue)) {
          (output as any)[key] = this.deepMerge(targetValue, sourceValue);
        } else {
          (output as any)[key] = sourceValue;
        }
      }
    }
    return output;
  }
  
  protected isObject(item: unknown): item is Record<string, unknown> {
    return item !== null && typeof item === 'object' && !Array.isArray(item);
  }
}
