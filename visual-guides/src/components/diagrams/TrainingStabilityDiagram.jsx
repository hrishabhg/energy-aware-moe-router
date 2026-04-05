import React, { useState, useMemo } from 'react';
import {
  Paper,
  Card,
  CardContent,
  CardHeader,
  Switch,
  Box,
  Typography,
  Grid,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';

/**
 * Generates a smooth loss curve with optional instability spike and solutions
 */
const generateLossCurve = (steps, hasProblem = true, solutions = {}) => {
  const data = [];
  const solutionActive = Object.values(solutions).some((v) => v);

  for (let i = 0; i <= steps; i++) {
    let loss;

    // Normal decreasing curve
    const baseLoss = 5.0 * Math.exp(-i / 2000);

    if (!hasProblem || solutionActive) {
      // Solutions applied or no problem - smooth curve
      loss = baseLoss + (Math.random() * 0.1 - 0.05);
    } else {
      // Instability spike at step 5000
      if (i < 5000) {
        loss = baseLoss + (Math.random() * 0.1 - 0.05);
      } else if (i < 5500) {
        const spikeIntensity = Math.sin((i - 5000) / 500 * Math.PI);
        loss = baseLoss + 3.5 * spikeIntensity + (Math.random() * 0.2 - 0.1);
      } else {
        loss = baseLoss + (Math.random() * 0.1 - 0.05);
      }
    }

    // Apply solution dampening
    if (solutions.selectivePrecision) {
      loss *= 0.85;
    }
    if (solutions.smallerInit) {
      loss *= 0.9;
    }
    if (solutions.expertDropout) {
      loss *= 0.92;
    }

    data.push({ step: i, loss: Math.max(0.01, loss) });
  }

  return data;
};

/**
 * SVG Line Chart Component
 */
const LossChart = ({ data, theme }) => {
  if (!data || data.length === 0) return null;

  const width = 800;
  const height = 300;
  const padding = 40;

  const maxLoss = Math.max(...data.map((d) => d.loss));
  const maxStep = Math.max(...data.map((d) => d.step));

  const xScale = (step) => (step / maxStep) * (width - 2 * padding) + padding;
  const yScale = (loss) => height - padding - (loss / maxLoss) * (height - 2 * padding);

  const pathData = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(d.step)} ${yScale(d.loss)}`)
    .join(' ');

  return (
    <motion.svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map((frac) => (
        <line
          key={`hgrid-${frac}`}
          x1={padding}
          y1={yScale(maxLoss * frac)}
          x2={width - padding}
          y2={yScale(maxLoss * frac)}
          stroke={theme.palette.divider}
          strokeDasharray="4"
          strokeWidth="1"
        />
      ))}

      {/* Axes */}
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke={theme.palette.text.primary}
        strokeWidth="2"
      />
      <line
        x1={padding}
        y1={padding}
        x2={padding}
        y2={height - padding}
        stroke={theme.palette.text.primary}
        strokeWidth="2"
      />

      {/* Axis labels */}
      <text
        x={width / 2}
        y={height - 5}
        textAnchor="middle"
        fill={theme.palette.text.primary}
        fontSize="12"
      >
        Training Steps
      </text>
      <text
        x={15}
        y={height / 2}
        textAnchor="middle"
        fill={theme.palette.text.primary}
        fontSize="12"
        transform={`rotate(-90 15 ${height / 2})`}
      >
        Loss
      </text>

      {/* Step labels */}
      {[0, 0.5, 1].map((frac) => {
        const step = Math.round(maxStep * frac);
        return (
          <text
            key={`step-${frac}`}
            x={xScale(step)}
            y={height - padding + 20}
            textAnchor="middle"
            fill={theme.palette.text.secondary}
            fontSize="11"
          >
            {step}
          </text>
        );
      })}

      {/* Loss labels */}
      {[0.25, 0.5, 0.75].map((frac) => {
        const loss = (maxLoss * frac).toFixed(1);
        return (
          <text
            key={`loss-${frac}`}
            x={padding - 10}
            y={yScale(maxLoss * frac) + 4}
            textAnchor="end"
            fill={theme.palette.text.secondary}
            fontSize="11"
          >
            {loss}
          </text>
        );
      })}

      {/* Loss curve */}
      <path
        d={pathData}
        fill="none"
        stroke={theme.palette.primary.main}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Spike annotation (if instability visible) */}
      {data[5000]?.loss > 2 && (
        <>
          <circle cx={xScale(5000)} cy={yScale(data[5000].loss)} r="5" fill={theme.palette.error.main} />
          <line
            x1={xScale(5000)}
            y1={yScale(data[5000].loss)}
            x2={xScale(5000) + 100}
            y2={yScale(data[5000].loss) - 80}
            stroke={theme.palette.error.main}
            strokeWidth="2"
          />
          <text
            x={xScale(5000) + 105}
            y={yScale(data[5000].loss) - 70}
            fill={theme.palette.error.main}
            fontSize="12"
            fontWeight="bold"
          >
            Instability Spike
          </text>
          <text
            x={xScale(5000) + 105}
            y={yScale(data[5000].loss) - 55}
            fill={theme.palette.error.main}
            fontSize="11"
          >
            Router logits diverge
          </text>
        </>
      )}
    </motion.svg>
  );
};

/**
 * Data type visualization for Selective Precision
 */
const DataTypeViz = ({ theme }) => {
  return (
    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 1 }}>
      <Box
        sx={{
          flex: 4,
          height: 30,
          bgcolor: theme.palette.info.light,
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          fontWeight: 'bold',
          color: theme.palette.info.dark,
        }}
      >
        BFloat16
      </Box>
      <Box
        sx={{
          flex: 1,
          height: 30,
          bgcolor: theme.palette.warning.light,
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          fontWeight: 'bold',
          color: theme.palette.warning.dark,
        }}
      >
        FP32
      </Box>
    </Box>
  );
};

/**
 * Initialization scale visualization
 */
const InitScaleViz = ({ theme }) => {
  return (
    <Box sx={{ mt: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="caption">Standard (σ=0.02)</Typography>
        <Typography variant="caption">Reduced (σ=0.002)</Typography>
      </Box>
      <Box sx={{ display: 'flex', gap: 0.5, height: 20 }}>
        <Box
          sx={{
            flex: 1,
            bgcolor: theme.palette.error.light,
            borderRadius: 0.5,
          }}
        />
        <Box
          sx={{
            flex: 1,
            bgcolor: theme.palette.warning.light,
            borderRadius: 0.5,
          }}
        />
        <Box
          sx={{
            flex: 1,
            bgcolor: theme.palette.success.light,
            borderRadius: 0.5,
          }}
        />
      </Box>
    </Box>
  );
};

/**
 * Expert grid visualization with dropout
 */
const ExpertGridViz = ({ active, theme }) => {
  const experts = Array.from({ length: 8 }, (_, i) => i);

  return (
    <Box sx={{ mt: 1 }}>
      <Grid container spacing={0.5}>
        {experts.map((idx) => {
          const isDropped = active && idx % 3 === 0;
          return (
            <Grid item xs={3} key={idx}>
              <motion.div
                animate={{ opacity: isDropped ? 0.4 : 1 }}
                transition={{ duration: 0.3 }}
              >
                <Box
                  sx={{
                    width: '100%',
                    paddingBottom: '100%',
                    position: 'relative',
                    bgcolor: isDropped ? theme.palette.action.disabled : theme.palette.success.light,
                    borderRadius: 1,
                  }}
                >
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      color: isDropped ? theme.palette.text.disabled : theme.palette.success.dark,
                    }}
                  >
                    {isDropped ? '✗' : 'E'}
                  </Box>
                </Box>
              </motion.div>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

/**
 * Main TrainingStabilityDiagram Component
 */
const TrainingStabilityDiagram = () => {
  const theme = useTheme();
  const [solutions, setSolutions] = useState({
    selectivePrecision: false,
    smallerInit: false,
    expertDropout: false,
  });

  const lossData = useMemo(() => generateLossCurve(10000, true, solutions), [solutions]);

  const handleSolutionToggle = (key) => {
    setSolutions((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <Box sx={{ p: 3, bgcolor: theme.palette.background.default }}>
      {/* Title */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <Typography variant="h4" gutterBottom sx={{ mb: 4, fontWeight: 'bold' }}>
          Training Stability in Mixture of Experts
        </Typography>
      </motion.div>

      {/* Section 1: The Instability Problem */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Paper elevation={3} sx={{ p: 3, mb: 4, bgcolor: theme.palette.background.paper }}>
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 2 }}>
            Section 1: The Instability Problem
          </Typography>

          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              The router's softmax computation can cause gradient explosion when expert logits diverge, leading to
              training collapse. Observe the loss spike at step 5000:
            </Typography>
            <Typography variant="caption" sx={{ bgcolor: theme.palette.error.light, p: 1, borderRadius: 1, display: 'block' }}>
              Router logits diverge → expert receives extreme gradients → training collapses
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'center', overflow: 'auto' }}>
            <LossChart data={lossData} theme={theme} />
          </Box>
        </Paper>
      </motion.div>

      {/* Section 2: Solutions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
          Section 2: Three Solutions
        </Typography>

        <Grid container spacing={3}>
          {/* Solution 1: Selective Precision */}
          <Grid item xs={12} md={6} lg={4}>
            <motion.div
              whileHover={{ y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <Card
                sx={{
                  height: '100%',
                  border: solutions.selectivePrecision ? `2px solid ${theme.palette.success.main}` : 'none',
                  transition: 'all 0.3s ease',
                }}
              >
                <CardHeader
                  title="Selective Precision"
                  subheader="BFloat16 + Float32 Routing"
                  action={
                    <Switch
                      checked={solutions.selectivePrecision}
                      onChange={() => handleSolutionToggle('selectivePrecision')}
                    />
                  }
                />
                <CardContent>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Router probabilities need higher precision because they make discrete routing decisions. Small
                    numerical errors can flip which expert a token goes to.
                  </Typography>
                  <DataTypeViz theme={theme} />
                  <Box
                    sx={{
                      mt: 2,
                      p: 1,
                      bgcolor: theme.palette.info.light,
                      borderRadius: 1,
                      fontSize: '12px',
                    }}
                  >
                    Most of model in BFloat16, router computation in Float32
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          {/* Solution 2: Smaller Initialization */}
          <Grid item xs={12} md={6} lg={4}>
            <motion.div
              whileHover={{ y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <Card
                sx={{
                  height: '100%',
                  border: solutions.smallerInit ? `2px solid ${theme.palette.success.main}` : 'none',
                  transition: 'all 0.3s ease',
                }}
              >
                <CardHeader
                  title="Smaller Initialization"
                  subheader="Reduced Router Weights"
                  action={
                    <Switch
                      checked={solutions.smallerInit}
                      onChange={() => handleSolutionToggle('smallerInit')}
                    />
                  }
                />
                <CardContent>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Smaller initial weights mean the router starts with more uniform probabilities, preventing early
                    expert collapse.
                  </Typography>
                  <InitScaleViz theme={theme} />
                  <Box
                    sx={{
                      mt: 2,
                      p: 1,
                      bgcolor: theme.palette.warning.light,
                      borderRadius: 1,
                      fontSize: '12px',
                    }}
                  >
                    σ reduced from 0.02 to 0.002 × 0.1
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          {/* Solution 3: Expert Dropout */}
          <Grid item xs={12} md={6} lg={4}>
            <motion.div
              whileHover={{ y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <Card
                sx={{
                  height: '100%',
                  border: solutions.expertDropout ? `2px solid ${theme.palette.success.main}` : 'none',
                  transition: 'all 0.3s ease',
                }}
              >
                <CardHeader
                  title="Expert Dropout"
                  subheader="Increased During Fine-tuning"
                  action={
                    <Switch
                      checked={solutions.expertDropout}
                      onChange={() => handleSolutionToggle('expertDropout')}
                    />
                  }
                />
                <CardContent>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Increasing dropout within expert FFNs from 0.1 to 0.4 during fine-tuning prevents overfitting to
                    routing patterns.
                  </Typography>
                  <ExpertGridViz active={solutions.expertDropout} theme={theme} />
                  <Box
                    sx={{
                      mt: 2,
                      p: 1,
                      bgcolor: theme.palette.success.light,
                      borderRadius: 1,
                      fontSize: '12px',
                    }}
                  >
                    Dropout rate: 0.1 → 0.4
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>

        {/* Summary */}
        {Object.values(solutions).some((v) => v) && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Paper
              elevation={2}
              sx={{
                mt: 4,
                p: 2,
                bgcolor: theme.palette.success.light,
                borderLeft: `4px solid ${theme.palette.success.main}`,
              }}
            >
              <Typography variant="body2" sx={{ color: theme.palette.success.dark }}>
                <strong>Combined Effect:</strong> When all three solutions are enabled, the loss curve becomes smooth and
                stable throughout training. These stabilization techniques are essential for scaling Mixture of Experts
                models to production.
              </Typography>
            </Paper>
          </motion.div>
        )}
      </motion.div>
    </Box>
  );
};

export default TrainingStabilityDiagram;
