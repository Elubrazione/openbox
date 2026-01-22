export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

export function generateGradientColors(
  startColor: string,
  endColor: string,
  steps: number
): string[] {
  const start = hexToRgb(startColor);
  const end = hexToRgb(endColor);
  
  if (!start || !end) return Array(steps).fill(startColor);
  
  return Array.from({ length: steps }, (_, i) => {
    const ratio = i / (steps - 1);
    const r = Math.round(start.r + (end.r - start.r) * ratio);
    const g = Math.round(start.g + (end.g - start.g) * ratio);
    const b = Math.round(start.b + (end.b - start.b) * ratio);
    return `rgb(${r}, ${g}, ${b})`;
  });
}

export function generateOpacityGradient(
  baseColor: string,
  steps: number,
  startOpacity = 0.4,
  endOpacity = 1
): string[] {
  const rgb = hexToRgb(baseColor);
  if (!rgb) return Array(steps).fill(baseColor);
  
  return Array.from({ length: steps }, (_, i) => {
    const opacity = startOpacity + (endOpacity - startOpacity) * (i / (steps - 1));
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
  });
}
