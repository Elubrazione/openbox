import type { ColorPalette, GridConfig, ChartTheme } from '../types';

// ============================================
// ============================================

export const lightPalette: ColorPalette = {
  primary: '#409EFF',
  secondary: '#909399',
  success: '#67C23A',
  warning: '#E6A23C',
  danger: '#F56C6C',
  info: '#909399',
  gradient: ['#409EFF', '#67C23A'],
  series: [
    '#5470c6', '#91cc75', '#fac858', '#ee6666',
    '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
    '#ea7ccc', '#48b8d0',
  ],
};

export const darkPalette: ColorPalette = {
  primary: '#409EFF',
  secondary: '#A3A6AD',
  success: '#85CE61',
  warning: '#EBB563',
  danger: '#F78989',
  info: '#A6A9AD',
  gradient: ['#409EFF', '#85CE61'],
  series: [
    '#4992ff', '#7cffb2', '#fddd60', '#ff6e76',
    '#58d9f9', '#05c091', '#ff8a45', '#8d48e3',
    '#dd79ff', '#37a2da',
  ],
};

export const brandPalette: ColorPalette = {
  primary: '#6366F1',
  secondary: '#8B5CF6',
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#6B7280',
  gradient: ['#6366F1', '#8B5CF6'],
  series: [
    '#6366F1', '#8B5CF6', '#EC4899', '#F43F5E',
    '#10B981', '#14B8A6', '#06B6D4', '#0EA5E9',
    '#F59E0B', '#84CC16',
  ],
};

export function getPalette(theme: ChartTheme): ColorPalette {
  switch (theme) {
    case 'dark':
      return darkPalette;
    case 'brand':
      return brandPalette;
    default:
      return lightPalette;
  }
}

// ============================================
// ============================================

export function getCompressionRatioColor(ratio: number): string {
  if (ratio > 0.9) return '#f56c6c';
  if (ratio > 0.7) return '#e6a23c';
  if (ratio > 0.5) return '#f0a020';
  return '#67c23a';
}

export const compressionRatioLegend = [
  { threshold: 0.5, color: '#67c23a', label: '<50% (Optimal)' },
  { threshold: 0.7, color: '#f0a020', label: '50-70% (Good)' },
  { threshold: 0.9, color: '#e6a23c', label: '70-90% (Moderate)' },
  { threshold: 1.0, color: '#f56c6c', label: '>90% (High)' },
];

// ============================================
// ============================================

export const defaultGrid: GridConfig = {
  left: '5%',
  right: '5%',
  top: '15%',
  bottom: '10%',
  containLabel: true,
};

export const compactGrid: GridConfig = {
  left: '3%',
  right: '3%',
  top: '10%',
  bottom: '5%',
  containLabel: true,
};

export const spaciousGrid: GridConfig = {
  left: '12%',
  right: '8%',
  top: '15%',
  bottom: '25%',
  containLabel: true,
};

export const defaultTitleStyle = {
  fontWeight: 'bold' as const,
  fontSize: 16,
  color: '#333',
};

export const defaultTooltip = {
  trigger: 'axis' as const,
  axisPointer: {
    type: 'shadow' as const,
  },
  backgroundColor: 'rgba(255, 255, 255, 0.95)',
  borderColor: '#e0e0e0',
  borderWidth: 1,
  textStyle: {
    color: '#333',
  },
};

export const defaultAnimation = {
  animation: true,
  animationDuration: 500,
  animationEasing: 'cubicOut' as const,
  animationDelay: 0,
};

export const noAnimation = {
  animation: false,
};

// ============================================
// ============================================

export const chartHeights = {
  small: '250px',
  medium: '350px',
  large: '500px',
  xlarge: '600px',
  auto: (itemCount: number, itemHeight = 25, minHeight = 400) => 
    `${Math.max(minHeight, itemCount * itemHeight)}px`,
};
