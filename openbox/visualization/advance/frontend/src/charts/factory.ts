import type { EChartsOption } from 'echarts';
import type { 
  IChartConfigFactory, 
  IChartStrategy, 
  ChartType, 
  ChartData,
  ChartConfigContext,
  StrategyRegistry,
  ChartPreset,
  PresetName,
} from './types';
import {
  BarChartStrategy,
  HorizontalBarChartStrategy,
  CompressionRatioBarStrategy,
  LineChartStrategy,
  DimensionEvolutionStrategy,
  HeatmapChartStrategy,
  MultiTaskHeatmapStrategy,
} from './strategies';
import { chartPresets } from './config';

class ChartConfigFactory implements IChartConfigFactory {
  private static instance: ChartConfigFactory;
  private strategies: StrategyRegistry = new Map();
  
  private constructor() {
    this.registerDefaultStrategies();
  }
  
  static getInstance(): ChartConfigFactory {
    if (!ChartConfigFactory.instance) {
      ChartConfigFactory.instance = new ChartConfigFactory();
    }
    return ChartConfigFactory.instance;
  }
  
  private registerDefaultStrategies(): void {
    this.registerStrategy(new BarChartStrategy());
    this.registerStrategy(new HorizontalBarChartStrategy());
    this.registerStrategy(new LineChartStrategy());
    this.registerStrategy(new HeatmapChartStrategy());
  }
  
  registerStrategy(strategy: IChartStrategy): void {
    this.strategies.set(strategy.type, strategy);
  }
  
  getStrategy(type: ChartType): IChartStrategy | undefined {
    return this.strategies.get(type);
  }
  
  createConfig<T extends ChartData>(
    type: ChartType,
    data: T,
    options?: Partial<ChartConfigContext>
  ): EChartsOption {
    const strategy = this.strategies.get(type);
    
    if (!strategy) {
      console.warn(`No strategy found for chart type: ${type}`);
      return {};
    }
    
    if (!strategy.validateData(data)) {
      console.warn(`Invalid data for chart type: ${type}`);
      return {};
    }
    
    const context: ChartConfigContext = {
      type,
      theme: options?.theme || 'light',
      ...strategy.getDefaultContext(),
      ...options,
    };
    
    return strategy.createOption(data, context);
  }
  
  createFromPreset<T extends ChartData>(
    presetName: PresetName,
    data: T,
    overrides?: Partial<ChartConfigContext>
  ): EChartsOption {
    const preset = chartPresets[presetName];
    
    if (!preset) {
      console.warn(`No preset found: ${presetName}`);
      return this.createConfig('bar', data, overrides);
    }
    
    return this.createConfig(preset.type, data, {
      ...preset.context,
      ...overrides,
    });
  }
  
  getAvailableTypes(): ChartType[] {
    return Array.from(this.strategies.keys());
  }
}
export const getChartFactory = () => ChartConfigFactory.getInstance();

export function createChartConfig<T extends ChartData>(
  type: ChartType,
  data: T,
  options?: Partial<ChartConfigContext>
): EChartsOption {
  return getChartFactory().createConfig(type, data, options);
}

export function createFromPreset<T extends ChartData>(
  presetName: PresetName,
  data: T,
  overrides?: Partial<ChartConfigContext>
): EChartsOption {
  return getChartFactory().createFromPreset(presetName, data, overrides);
}
export const createCompressionRatioStrategy = () => new CompressionRatioBarStrategy();
export const createDimensionEvolutionStrategy = () => new DimensionEvolutionStrategy();
export const createMultiTaskHeatmapStrategy = () => new MultiTaskHeatmapStrategy();
