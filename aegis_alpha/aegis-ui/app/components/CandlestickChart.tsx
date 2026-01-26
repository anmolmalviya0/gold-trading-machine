'use client';

import { useEffect, useRef } from 'react';
import { createChart, ColorType, ISeriesApi, UTCTimestamp } from 'lightweight-charts';

interface CandlestickChartProps {
    data: Array<{
        time: string | number;
        open: number;
        high: number;
        low: number;
        close: number;
    }>;
}

export default function CandlestickChart({ data }: CandlestickChartProps) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
    const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;
        
        // Clean up existing chart
        if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
        }

        // Create new chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: 600,
            layout: {
                background: { type: ColorType.Solid, color: '#0A0A0A' }, // Solid dark BG to stop flickering
                textColor: '#525252',
            },
            grid: {
                vertLines: { visible: false }, // Cleaner look
                horzLines: { color: 'rgba(16, 185, 129, 0.05)' },
            },
            crosshair: {
                mode: 1, // Magnet mode
                vertLine: {
                    color: 'rgba(16, 185, 129, 0.4)',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#10b981',
                },
                horzLine: {
                    color: 'rgba(16, 185, 129, 0.4)',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#10b981',
                },
            },
            rightPriceScale: {
                borderColor: 'rgba(16, 185, 129, 0.1)',
                scaleMargins: {
                    top: 0.2,
                    bottom: 0.2,
                },
            },
            timeScale: {
                borderColor: 'rgba(16, 185, 129, 0.1)',
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 12,
                barSpacing: 10,
                minBarSpacing: 5,
                fixLeftEdge: true,
                fixRightEdge: true,
            },
        });

        chartRef.current = chart;

        // Add Candlestick Series with proper styling
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });
        
        seriesRef.current = candleSeries;

        // Responsive resize
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
        };
    }, []);

    // Update data when it changes
    // Update data when it changes
    useEffect(() => {
        if (!seriesRef.current || !data || data.length === 0) return;

        try {
            // Transform data - Handle both String (old) and Number (new) formats
            const chartData = data
                .map(candle => {
                    let timestamp: number;
                    
                    // If backend sends raw number (UNIX seconds), use it directly
                    if (typeof candle.time === 'number') {
                        timestamp = candle.time;
                    } else {
                        // Fallback for string parsing
                        timestamp = new Date(candle.time).getTime() / 1000;
                    }
                    
                    if (isNaN(timestamp) || !candle.open || !candle.high || !candle.low || !candle.close) {
                        return null;
                    }
                    
                    return {
                        time: timestamp as UTCTimestamp,
                        open: candle.open,
                        high: candle.high,
                        low: candle.low,
                        close: candle.close,
                    };
                })
                .filter((d): d is NonNullable<typeof d> => d !== null);
            
            // Sort by timestamp (ascending)
            chartData.sort((a, b) => (a.time as number) - (b.time as number));
            
            // Remove duplicates
            const uniqueData = chartData.filter((item, index, self) => 
                index === 0 || item.time !== self[index - 1].time
            );
            
            if (uniqueData.length > 0) {
                seriesRef.current.setData(uniqueData);
                
                // Only fit once on initial data load, not on every update
                // This prevents the "Auto-Zoom-Out" bug requested by the user.
                const isInitialLoad = !chartRef.current?.timeScale()?.getVisibleRange();
                if (isInitialLoad) {
                    chartRef.current?.timeScale()?.fitContent();
                }
            }
        } catch (e) {
            console.error('Chart data error:', e);
        }
    }, [data]);

    return (
        <div 
            ref={chartContainerRef} 
            className="w-full h-[600px] relative"
            style={{ minHeight: '600px' }}
        />
    );
}
