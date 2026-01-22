import type { EChartsOption } from 'echarts';

export interface LazyChartProps {
  
  data: any[];
  
  type: 'line' | 'bar' | 'scatter' | 'heatmap' | 'pie';
  
  getOption?: (data: any[]) => EChartsOption;
  
  option?: EChartsOption;
  
  height?: number | string;
  
  width?: number | string;
  
  title?: string;
  
  chunkSize?: number;
  
  chunkThreshold?: number;
  
  lazyThreshold?: number;
  
  lazyRootMargin?: string;
  
  disableLazy?: boolean;
  
  disableChunking?: boolean;
  
  onLoadComplete?: () => void;
  
  placeholder?: React.ReactNode;
  
  className?: string;
  
  style?: React.CSSProperties;
  
  showProgress?: boolean;
  
  notMerge?: boolean;
  
  lazyUpdate?: boolean;
}

export interface ChartPlaceholderProps {
  
  height?: number | string;
  
  width?: number | string;
  
  title?: string;
  
  animate?: boolean;
  
  className?: string;
  
  style?: React.CSSProperties;
}

export interface LoadingBarProps {
  
  progress: number;
  
  height?: number;
  
  showText?: boolean;
  
  color?: string;
  
  backgroundColor?: string;
  
  className?: string;
  
  style?: React.CSSProperties;
  
  dataInfo?: {
    loaded: number;
    total: number;
  };
}
