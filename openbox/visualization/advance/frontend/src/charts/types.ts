import type { EChartsOption } from 'echarts';

// ============================================
// ============================================

export type ChartType = 
  | 'bar'
  | 'horizontalBar'
  | 'line'
  | 'pie'
  | 'heatmap'
  | 'custom'
  | 'scatter';

export type ChartTheme = 'light' | 'dark' | 'brand';

export interface ColorPalette {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  danger: string;
  info: string;
  gradient: [string, string];
  series: string[];
}

export interface ChartDimensions {
  width?: string | number;
  height: string | number;
  minHeight?: number;
  aspectRatio?: number;
}

export interface GridConfig {
  left: string | number;
  right: string | number;
  top: string | number;
  bottom: string | number;
  containLabel?: boolean;
}

export interface TitleConfig {
  text: string;
  subtext?: string;
  left?: string | number;
  top?: string | number;
  textStyle?: {
    fontWeight?: string | number;
    fontSize?: number;
    color?: string;
  };
}

export interface TooltipConfig {
  trigger?: 'item' | 'axis' | 'none';
  formatter?: string | ((params: any) => string);
  axisPointer?: {
    type?: 'line' | 'shadow' | 'cross' | 'none';
  };
}

export interface LegendConfig {
  show?: boolean;
  data?: string[];
  position?: 'top' | 'bottom' | 'left' | 'right';
  orient?: 'horizontal' | 'vertical';
}

export interface ChartConfigContext {
  type: ChartType;
  theme: ChartTheme;
  title?: TitleConfig;
  tooltip?: TooltipConfig;
  legend?: LegendConfig;
  grid?: GridConfig;
  dimensions?: ChartDimensions;
  animation?: boolean;
  colors?: ColorPalette;
}

// ============================================
// ============================================

export interface DataItem {
  name: string;
  value: number | number[];
  itemStyle?: { color?: string; opacity?: number };
  label?: { show?: boolean; formatter?: string | ((params: any) => string) };
}

export interface BarChartData {
  categories: string[];
  values: number[];
  colors?: string[];
  labels?: boolean;
}

export interface LineChartData {
  xData: (string | number)[];
  yData: number[];
  smooth?: boolean;
  areaStyle?: boolean;
}

export interface PieChartData {
  items: Array<{ name: string; value: number; color?: string }>;
  radius?: [string, string] | string;
}

export interface HeatmapChartData {
  xCategories: string[];
  yCategories: string[];
  values: number[][];
  min?: number;
  max?: number;
}

export interface CustomChartData {
  series: any[];
  rawData?: any;
}

export type ChartData = 
  | BarChartData 
  | LineChartData 
  | PieChartData 
  | HeatmapChartData 
  | CustomChartData;

// ============================================
// ============================================

export interface IChartStrategy<T extends ChartData = ChartData> {
  readonly type: ChartType;
  createOption(data: T, context: ChartConfigContext): EChartsOption;
  validateData(data: unknown): data is T;
  getDefaultContext(): Partial<ChartConfigContext>;
}

export type StrategyRegistry = Map<ChartType, IChartStrategy>;

export interface IChartConfigFactory {
  registerStrategy(strategy: IChartStrategy): void;
  createConfig<T extends ChartData>(type: ChartType, data: T, options?: Partial<ChartConfigContext>): EChartsOption;
  getAvailableTypes(): ChartType[];
}

// ============================================
// ============================================

export interface UseChartConfigParams<T extends ChartData = ChartData> {
  type: ChartType;
  data: T;
  options?: Partial<ChartConfigContext>;
  deps?: any[];
}

export interface UseChartConfigReturn {
  option: EChartsOption;
  dimensions: ChartDimensions;
  isValid: boolean;
  error: string | null;
}

// ============================================
// ============================================

export interface ChartPreset {
  name: string;
  type: ChartType;
  context: Partial<ChartConfigContext>;
  description?: string;
}

export type PresetName = 
  | 'parameterImportance'
  | 'dimensionReduction'
  | 'compressionRatio'
  | 'rangeCompression'
  | 'dimensionEvolution'
  | 'multiTaskHeatmap';
