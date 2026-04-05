import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
  Grid,
  Chip,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';

const ScalingParallelismDiagram = () => {
  const theme = useTheme();
  const [parallelismMode, setParallelismMode] = useState('data');
  const [lineProgress, setLineProgress] = useState(0);
  const expertColors = theme.palette.expert?.colors || [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
  ];

  // Animate line drawing on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setLineProgress(1);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  // Data points for scaling efficiency curves
  const denseData = [
    { x: 1, y: 0.3 },
    { x: 2, y: 0.5 },
    { x: 4, y: 0.65 },
    { x: 8, y: 0.75 },
    { x: 16, y: 0.82 },
  ];

  const moeTop2Data = [
    { x: 1, y: 0.4 },
    { x: 2, y: 0.6 },
    { x: 4, y: 0.73 },
    { x: 8, y: 0.82 },
    { x: 16, y: 0.88 },
  ];

  const switchData = [
    { x: 1, y: 0.5 },
    { x: 2, y: 0.68 },
    { x: 4, y: 0.8 },
    { x: 8, y: 0.88 },
    { x: 16, y: 0.93 },
  ];

  // Convert data to SVG path
  const generatePath = (data, chartWidth = 400, chartHeight = 250) => {
    const padding = 40;
    const maxX = 16;
    const maxY = 1;

    const points = data.map((point) => {
      const px = padding + (point.x / maxX) * (chartWidth - 2 * padding);
      const py = chartHeight - padding - (point.y / maxY) * (chartHeight - 2 * padding);
      return `${px},${py}`;
    });

    return `M ${points.join(' L ')}`;
  };

  // Smooth curve generation using quadratic Bezier
  const generateSmoothPath = (data, chartWidth = 400, chartHeight = 250) => {
    const padding = 40;
    const maxX = 16;
    const maxY = 1;

    const points = data.map((point) => ({
      x: padding + (point.x / maxX) * (chartWidth - 2 * padding),
      y: chartHeight - padding - (point.y / maxY) * (chartHeight - 2 * padding),
    }));

    if (points.length < 2) return '';

    let path = `M ${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const next = points[i + 1] || curr;

      const cpx = (curr.x + next.x) / 2;
      const cpy = (curr.y + next.y) / 2;

      path += ` Q ${curr.x},${curr.y} ${cpx},${cpy}`;
    }
    path += ` L ${points[points.length - 1].x},${points[points.length - 1].y}`;

    return path;
  };

  // ============ SECTION 1: SCALING EFFICIENCY ============
  const ScalingEfficiencyChart = () => {
    const chartWidth = 450;
    const chartHeight = 280;
    const padding = 50;

    return (
      <Paper
        elevation={2}
        sx={{
          p: 3,
          mb: 4,
          bgcolor: theme.palette.mode === 'dark' ? 'grey.900' : 'background.paper',
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Scaling Efficiency
        </Typography>

        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <svg width={chartWidth} height={chartHeight} style={{ border: `1px solid ${theme.palette.divider}` }}>
            {/* X Axis */}
            <line
              x1={padding}
              y1={chartHeight - padding}
              x2={chartWidth - 20}
              y2={chartHeight - padding}
              stroke={theme.palette.text.secondary}
              strokeWidth="2"
            />

            {/* Y Axis */}
            <line
              x1={padding}
              y1={20}
              x2={padding}
              y2={chartHeight - padding}
              stroke={theme.palette.text.secondary}
              strokeWidth="2"
            />

            {/* Grid lines */}
            {[1, 2, 4, 8, 16].map((val, idx) => {
              const x = padding + ((val / 16) * (chartWidth - 2 * padding - 20));
              return (
                <g key={`grid-x-${idx}`}>
                  <line
                    x1={x}
                    y1={20}
                    x2={x}
                    y2={chartHeight - padding}
                    stroke={theme.palette.divider}
                    strokeWidth="1"
                    strokeDasharray="4,4"
                    opacity="0.5"
                  />
                  <text
                    x={x}
                    y={chartHeight - padding + 20}
                    textAnchor="middle"
                    fontSize="12"
                    fill={theme.palette.text.secondary}
                  >
                    {val}x
                  </text>
                </g>
              );
            })}

            {[0.2, 0.4, 0.6, 0.8, 1.0].map((val, idx) => {
              const y = chartHeight - padding - val * (chartHeight - 2 * padding);
              return (
                <g key={`grid-y-${idx}`}>
                  <line
                    x1={padding}
                    y1={y}
                    x2={chartWidth - 20}
                    y2={y}
                    stroke={theme.palette.divider}
                    strokeWidth="1"
                    strokeDasharray="4,4"
                    opacity="0.5"
                  />
                  <text
                    x={padding - 10}
                    y={y + 4}
                    textAnchor="end"
                    fontSize="12"
                    fill={theme.palette.text.secondary}
                  >
                    {val.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Axis labels */}
            <text
              x={chartWidth / 2}
              y={chartHeight - 5}
              textAnchor="middle"
              fontSize="13"
              fontWeight="600"
              fill={theme.palette.text.primary}
            >
              FLOPS (compute budget)
            </text>

            <text
              x={20}
              y={chartHeight / 2}
              textAnchor="middle"
              fontSize="13"
              fontWeight="600"
              fill={theme.palette.text.primary}
              transform={`rotate(-90 20 ${chartHeight / 2})`}
            >
              Negative Log Perplexity
            </text>

            {/* Dense T5 line */}
            <motion.path
              d={generateSmoothPath(denseData, chartWidth - 70, chartHeight - 70)}
              stroke={expertColors[0]}
              strokeWidth="3"
              fill="none"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: lineProgress }}
              transition={{ duration: 1.2, ease: 'easeInOut' }}
            />

            {/* MoE top-2 line */}
            <motion.path
              d={generateSmoothPath(moeTop2Data, chartWidth - 70, chartHeight - 70)}
              stroke={expertColors[2]}
              strokeWidth="3"
              fill="none"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: lineProgress }}
              transition={{ duration: 1.2, delay: 0.1, ease: 'easeInOut' }}
            />

            {/* Switch top-1 line */}
            <motion.path
              d={generateSmoothPath(switchData, chartWidth - 70, chartHeight - 70)}
              stroke={expertColors[4]}
              strokeWidth="3"
              fill="none"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: lineProgress }}
              transition={{ duration: 1.2, delay: 0.2, ease: 'easeInOut' }}
            />

            {/* Data point markers for Switch */}
            {switchData.map((point, idx) => {
              const px = padding + ((point.x / 16) * (chartWidth - 2 * padding - 20));
              const py = chartHeight - padding - (point.y * (chartHeight - 2 * padding));
              return (
                <motion.circle
                  key={`switch-marker-${idx}`}
                  cx={px}
                  cy={py}
                  r="4"
                  fill={expertColors[4]}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 1.4 + idx * 0.1 }}
                />
              );
            })}
          </svg>
        </Box>

        {/* Legend */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mb: 3, flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 20, height: 3, bgcolor: expertColors[0] }} />
            <Typography variant="body2">Dense T5</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 20, height: 3, bgcolor: expertColors[2] }} />
            <Typography variant="body2">MoE top-2</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 20, height: 3, bgcolor: expertColors[4] }} />
            <Typography variant="body2">Switch top-1</Typography>
          </Box>
        </Box>

        {/* Key Callouts */}
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Paper
              sx={{
                p: 2,
                bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.50',
                borderLeft: `4px solid ${expertColors[4]}`,
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                7× Less Compute
              </Typography>
              <Typography variant="body2">
                Switch Transformer reaches T5-Base quality at 7x less compute
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Paper
              sx={{
                p: 2,
                bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.50',
                borderLeft: `4px solid ${expertColors[4]}`,
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                1.6T Parameters
              </Typography>
              <Typography variant="body2">
                Switch-XXL has 1.6T parameters but trains at same cost as T5-XXL
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Paper>
    );
  };

  // ============ SECTION 2: PARALLELISM STRATEGIES ============
  const ParallelismGrid = () => {
    const numDevices = 4;

    const deviceBoxes = {
      data: [
        { device: 0, experts: [0, 1, 2, 3, 4, 5, 6, 7], label: 'All Experts' },
        { device: 1, experts: [0, 1, 2, 3, 4, 5, 6, 7], label: 'All Experts' },
        { device: 2, experts: [0, 1, 2, 3, 4, 5, 6, 7], label: 'All Experts' },
        { device: 3, experts: [0, 1, 2, 3, 4, 5, 6, 7], label: 'All Experts' },
      ],
      model: [
        { device: 0, experts: [0, 1], label: 'E0-E1' },
        { device: 1, experts: [2, 3], label: 'E2-E3' },
        { device: 2, experts: [4, 5], label: 'E4-E5' },
        { device: 3, experts: [6, 7], label: 'E6-E7' },
      ],
      combined: [
        { device: 0, experts: [0, 1, 2, 3], label: 'E0-E3' },
        { device: 1, experts: [0, 1, 2, 3], label: 'E0-E3' },
        { device: 2, experts: [4, 5, 6, 7], label: 'E4-E7' },
        { device: 3, experts: [4, 5, 6, 7], label: 'E4-E7' },
      ],
    };

    const communicationCosts = {
      data: 'Low',
      model: 'High',
      combined: 'Medium',
    };

    const communicationColors = {
      Low: theme.palette.success.main,
      Medium: theme.palette.warning.main,
      High: theme.palette.error.main,
    };

    const currentBoxes = deviceBoxes[parallelismMode];

    return (
      <Paper
        elevation={2}
        sx={{
          p: 3,
          bgcolor: theme.palette.mode === 'dark' ? 'grey.900' : 'background.paper',
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          Parallelism Strategies
        </Typography>

        {/* Mode Toggle */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3, gap: 2, flexWrap: 'wrap' }}>
          <ToggleButtonGroup
            value={parallelismMode}
            exclusive
            onChange={(e, newMode) => {
              if (newMode !== null) setParallelismMode(newMode);
            }}
            size="small"
          >
            <ToggleButton value="data">Data Parallelism</ToggleButton>
            <ToggleButton value="model">Expert Parallelism</ToggleButton>
            <ToggleButton value="combined">Combined</ToggleButton>
          </ToggleButtonGroup>

          <Chip
            label={`Communication: ${communicationCosts[parallelismMode]}`}
            color={
              communicationCosts[parallelismMode] === 'Low'
                ? 'success'
                : communicationCosts[parallelismMode] === 'Medium'
                  ? 'warning'
                  : 'error'
            }
            variant="outlined"
            size="small"
          />
        </Box>

        {/* Description */}
        <Box sx={{ mb: 3, p: 2, bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.50', borderRadius: 1 }}>
          <Typography variant="body2">
            {parallelismMode === 'data' &&
              'Each device has a full copy of all experts. Data is distributed across devices.'}
            {parallelismMode === 'model' &&
              'Experts are distributed across devices. All-to-all communication between devices.'}
            {parallelismMode === 'combined' &&
              'Groups of devices share experts with data parallelism within groups.'}
          </Typography>
        </Box>

        {/* Device Grid with Communication Arrows */}
        <Box sx={{ position: 'relative', mb: 4 }}>
          {parallelismMode === 'model' && (
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '200px',
                pointerEvents: 'none',
              }}
            >
              {/* All-to-all communication arrows */}
              {[0, 1, 2, 3].map((from) =>
                [0, 1, 2, 3]
                  .filter((to) => to !== from)
                  .map((to) => {
                    const fromX = 80 + from * 120;
                    const toX = 80 + to * 120;
                    const y = 100;
                    return (
                      <motion.line
                        key={`arrow-${from}-${to}`}
                        x1={fromX}
                        y1={y}
                        x2={toX}
                        y2={y}
                        stroke={theme.palette.error.main}
                        strokeWidth="1.5"
                        strokeDasharray="4,4"
                        opacity="0.4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.4 }}
                        transition={{ delay: 0.3 }}
                      />
                    );
                  }),
              )}
            </svg>
          )}

          <Grid container spacing={2} sx={{ position: 'relative', zIndex: 1 }}>
            <AnimatePresence mode="wait">
              {currentBoxes.map((box, idx) => (
                <Grid item xs={12} sm={6} md={3} key={`device-${box.device}`}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <Paper
                      sx={{
                        p: 2,
                        textAlign: 'center',
                        bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.50',
                        border: `2px solid ${theme.palette.divider}`,
                        minHeight: '220px',
                        display: 'flex',
                        flexDirection: 'column',
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                        Device {box.device}
                      </Typography>

                      {/* Expert boxes */}
                      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1, justifyContent: 'center' }}>
                        {box.experts.map((expert, eIdx) => (
                          <motion.div
                            key={`expert-${expert}`}
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ delay: 0.2 + eIdx * 0.05 }}
                          >
                            <Box
                              sx={{
                                p: 1,
                                bgcolor: expertColors[expert % expertColors.length],
                                borderRadius: 0.5,
                                color: 'white',
                                fontSize: '11px',
                                fontWeight: 600,
                              }}
                            >
                              E{expert}
                            </Box>
                          </motion.div>
                        ))}
                      </Box>

                      <Typography variant="caption" sx={{ mt: 1, color: theme.palette.text.secondary }}>
                        {box.label}
                      </Typography>

                      {parallelismMode === 'data' && (
                        <Typography variant="caption" sx={{ mt: 1, color: theme.palette.info.main }}>
                          Batch {box.device}
                        </Typography>
                      )}
                    </Paper>
                  </motion.div>
                </Grid>
              ))}
            </AnimatePresence>
          </Grid>
        </Box>

        {/* Mode Details */}
        <Paper
          sx={{
            p: 2,
            bgcolor: theme.palette.mode === 'dark' ? 'grey.800' : 'grey.50',
            borderRadius: 1,
          }}
        >
          <Grid container spacing={2}>
            <Grid item xs={12} sm={4}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Scaling
              </Typography>
              <Typography variant="body2">
                {parallelismMode === 'data' && 'Linear with number of devices'}
                {parallelismMode === 'model' && 'Expert parameters distributed'}
                {parallelismMode === 'combined' && 'Hybrid scaling'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Load Balancing
              </Typography>
              <Typography variant="body2">
                {parallelismMode === 'data' && 'Balanced per device'}
                {parallelismMode === 'model' && 'Per-expert load balancing'}
                {parallelismMode === 'combined' && 'Group-aware balancing'}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Best For
              </Typography>
              <Typography variant="body2">
                {parallelismMode === 'data' && 'Single machine or small cluster'}
                {parallelismMode === 'model' && 'Large distributed training'}
                {parallelismMode === 'combined' && 'Multi-machine training'}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Paper>
    );
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography
        variant="h5"
        sx={{
          mb: 3,
          fontWeight: 700,
          background: `linear-gradient(90deg, ${expertColors[0]}, ${expertColors[4]})`,
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Switch Transformer: Scaling & Parallelism
      </Typography>

      <ScalingEfficiencyChart />
      <ParallelismGrid />
    </Box>
  );
};

export default ScalingParallelismDiagram;
