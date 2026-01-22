import { useState, useEffect, useCallback } from 'react';
import apiService from '../services/api';
import { CompressionHistory, UseCompressionDataReturn } from '../types';

export const useCompressionData = (): UseCompressionDataReturn => {
  const [data, setData] = useState<CompressionHistory | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const history = await apiService.getCompressionHistory();
      setData(history);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load compression history';
      setError(message);
      console.error('useCompressionData error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchData,
    hasData: data !== null && data.history.length > 0,
  };
};

export default useCompressionData;
