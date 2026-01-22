/**
 * Common ECharts configuration utilities and best practices
 */

export interface ChartTheme {
  primaryColor: string;
  secondaryColor: string;
  successColor: string;
  warningColor: string;
  errorColor: string;
  textColor: string;
  backgroundColor: string;
}

export const defaultTheme: ChartTheme = {
  primaryColor: '#5470c6',
  secondaryColor: '#91cc75',
  successColor: '#67c23a',
  warningColor: '#e6a23c',
  errorColor: '#f56c6c',
  textColor: '#333',
  backgroundColor: '#fff',
};

/**
 * Get optimal grid configuration based on chart type
 */
export const getGridConfig = (type: 'horizontal-bar' | 'vertical-bar' | 'heatmap' | 'line') => {
  const configs = {
    'horizontal-bar': {
      left: '5%',
      right: '15%',
      top: '10%',
      bottom: '5%',
      containLabel: true,
    },
    'vertical-bar': {
      left: '10%',
      right: '8%',
      top: '15%',
      bottom: '25%',
      containLabel: true,
    },
    'heatmap': {
      left: '5%',
      right: '10%',
      top: '10%',
      bottom: '15%',
      containLabel: true,
    },
    'line': {
      left: '10%',
      right: '10%',
      top: '15%',
      bottom: '10%',
      containLabel: true,
    },
  };
  return configs[type];
};

/**
 * Common animation configuration
 */
export const animationConfig = {
  animation: true,
  animationDuration: 800,
  animationEasing: 'cubicOut' as const,
  animationDelay: (idx: number) => idx * 50,
};

/**
 * Common tooltip configuration
 */
export const getTooltipConfig = (trigger: 'axis' | 'item' = 'axis') => ({
  trigger,
  backgroundColor: 'rgba(50, 50, 50, 0.9)',
  borderColor: '#333',
  borderWidth: 0,
  padding: [10, 15],
  textStyle: {
    color: '#fff',
    fontSize: 12,
  },
  axisPointer: {
    type: trigger === 'axis' ? ('shadow' as const) : ('line' as const),
    shadowStyle: {
      color: 'rgba(150, 150, 150, 0.1)',
    },
  },
});

/**
 * Common title configuration
 */
export const getTitleConfig = (text: string, subtext?: string) => ({
  text,
  subtext,
  left: 'center',
  textStyle: {
    fontWeight: 'bold' as const,
    fontSize: 16,
    color: defaultTheme.textColor,
  },
  subtextStyle: {
    fontSize: 12,
    color: '#999',
  },
});

/**
 * Common legend configuration
 */
export const getLegendConfig = (position: 'top' | 'bottom' = 'top') => ({
  [position]: position === 'top' ? 30 : 0,
  left: 'center',
  textStyle: {
    fontSize: 12,
    color: defaultTheme.textColor,
  },
  itemWidth: 25,
  itemHeight: 14,
});

/**
 * Get color based on ratio/percentage
 */
export const getColorByRatio = (ratio: number): string => {
  if (ratio > 0.9) return defaultTheme.errorColor;
  if (ratio > 0.7) return defaultTheme.warningColor;
  if (ratio > 0.5) return '#f0a020';
  return defaultTheme.successColor;
};

/**
 * Format number with appropriate precision
 */
export const formatNumber = (value: number | null | undefined, decimals: number = 2): string => {
  if (value == null) return 'N/A';
  return value.toFixed(decimals);
};

/**
 * Truncate long text with ellipsis
 */
export const truncateText = (text: string, maxLength: number = 20): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

/**
 * Get shortened parameter name (last part after dot)
 */
export const getShortParamName = (fullName: string): string => {
  return fullName.split('.').pop() || fullName;
};

/**
 * Common responsive chart options
 */
export const responsiveOptions = {
  // Enable data zoom for large datasets
  dataZoom: {
    show: false,
    start: 0,
    end: 100,
  },
  // Enable toolbox for better UX
  toolbox: {
    show: false, // Can be enabled per chart as needed
    feature: {
      saveAsImage: { title: 'Save as Image' },
      dataView: { title: 'View Data', readOnly: true },
      restore: { title: 'Restore' },
    },
  },
};

/**
 * Enhanced tooltip formatter for parameter charts
 */
export const formatParamTooltip = (
  paramName: string,
  value: number | null | undefined,
  label: string = 'Value'
): string => {
  const valueStr = formatNumber(value, 4);
  return `<strong>${paramName}</strong><br/>${label}: ${valueStr}`;
};

/**
 * Enhanced tooltip formatter for range charts
 */
export const formatRangeTooltip = (
  paramName: string,
  min: number | null | undefined,
  max: number | null | undefined,
  type: 'original' | 'compressed' = 'original'
): string => {
  const minStr = formatNumber(min, 2);
  const maxStr = formatNumber(max, 2);
  const typeLabel = type === 'original' ? 'Original' : 'Compressed';
  return `<strong>${paramName}</strong><br/>${typeLabel}: [${minStr}, ${maxStr}]`;
};
