import React from 'react';
import { CompressionHistory } from '../../types';
import { useCompressionPipeline, useChartVisibility } from '../../hooks';
import RangeCompression from '../RangeCompression';

interface RangeCompressionContainerProps {
  data: CompressionHistory;
}

const RangeCompressionContainer: React.FC<RangeCompressionContainerProps> = ({ data }) => {
  const { activeSteps } = useCompressionPipeline(data);
  const { rangeCompressionSteps } = useChartVisibility(data);

  if (rangeCompressionSteps.length === 0) {
    return null;
  }

  return (
    <>
      {rangeCompressionSteps.map(({ step, index }) => (
        <section key={`range-${index}`} className="chart-section">
          <RangeCompression
            step={step}
            stepIndex={index + 1}
          />
        </section>
      ))}
    </>
  );
};

export default RangeCompressionContainer;
