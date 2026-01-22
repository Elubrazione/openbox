import * as echarts from 'echarts/core';
import { BarChart, LineChart, HeatmapChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
  DataZoomComponent,
} from 'echarts/components';
import { CanvasRenderer, SVGRenderer } from 'echarts/renderers';

echarts.use([
  BarChart,
  LineChart,
  HeatmapChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
  DataZoomComponent,
  CanvasRenderer,
  SVGRenderer,
]);

export default echarts;
export type { EChartsOption } from 'echarts';
