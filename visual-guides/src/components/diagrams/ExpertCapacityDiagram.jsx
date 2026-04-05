import React, { useState, useMemo } from 'react';
import {
  Paper,
  Slider,
  Box,
  Typography,
  useTheme,
  Stack,
  Card,
  CardContent,
} from '@mui/material';
import { motion } from 'framer-motion';

const ExpertCapacityDiagram = () => {
  const theme = useTheme();
  const [capacityFactor, setCapacityFactor] = useState(1.25);

  // Fixed token distribution: imbalanced across experts
  const tokenDistribution = [6, 3, 5, 2]; // Total: 16 tokens
  const totalTokens = 16;
  const numExperts = 4;

  // Compute expert capacity based on capacity factor
  const expertCapacity = useMemo(
    () => Math.ceil((totalTokens / numExperts) * capacityFactor),
    [capacityFactor]
  );

  // Get expert colors from theme
  const expertColors = theme.palette.expert?.colors || [
    '#FF6B6B',
    '#4ECDC4',
    '#45B7D1',
    '#FFA07A',
    '#98D8C8',
    '#F7DC6F',
    '#BB8FCE',
    '#85C1E2',
  ];

  // Compute dropped tokens and statistics
  const expertStats = useMemo(() => {
    return tokenDistribution.map((tokensAssigned, idx) => {
      const droppedTokens = Math.max(0, tokensAssigned - expertCapacity);
      const acceptedTokens = tokensAssigned - droppedTokens;
      return {
        expertIdx: idx,
        tokensAssigned,
        acceptedTokens,
        droppedTokens,
        capacityUtilization: (acceptedTokens / expertCapacity) * 100,
      };
    });
  }, [expertCapacity]);

  const totalDroppedTokens = expertStats.reduce(
    (sum, stat) => sum + stat.droppedTokens,
    0
  );

  return (
    <Box sx={{ width: '100%', p: 3 }}>
      <Stack spacing={4}>
        {/* Title */}
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
            Expert Capacity Factor Visualization
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Explore the tradeoff between capacity and compute efficiency
          </Typography>
        </Box>

        {/* Capacity Factor Slider */}
        <Paper elevation={2} sx={{ p: 3, backgroundColor: 'background.paper' }}>
          <Stack spacing={2}>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                Capacity Factor: {capacityFactor.toFixed(2)}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Determines max tokens per expert: (tokens_per_batch / num_experts) × capacity_factor
              </Typography>
            </Box>
            <Slider
              value={capacityFactor}
              onChange={(e, newValue) => setCapacityFactor(newValue)}
              min={0.5}
              max={2.0}
              step={0.1}
              marks={[
                { value: 0.5, label: '0.5 (Low)' },
                { value: 1.0, label: '1.0' },
                { value: 1.25, label: '1.25 (Default)' },
                { value: 2.0, label: '2.0 (High)' },
              ]}
              valueLabelDisplay="auto"
              valueLabelFormat={(val) => val.toFixed(2)}
              sx={{ mt: 2 }}
            />
          </Stack>
        </Paper>

        {/* Statistics Panel */}
        <Paper elevation={2} sx={{ p: 3, backgroundColor: 'background.paper' }}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3}>
            <StatBox label="Total Tokens" value={totalTokens} />
            <StatBox label="Num Experts" value={numExperts} />
            <StatBox label="Capacity per Expert" value={expertCapacity} />
            <StatBox
              label="Tokens Dropped"
              value={totalDroppedTokens}
              highlight={totalDroppedTokens > 0}
            />
          </Stack>
        </Paper>

        {/* Expert Buckets Visualization */}
        <Paper elevation={2} sx={{ p: 4, backgroundColor: 'background.paper' }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 3 }}>
            Expert Token Allocation
          </Typography>

          <Stack
            direction="row"
            spacing={2}
            sx={{
              justifyContent: 'space-around',
              alignItems: 'flex-end',
              minHeight: 350,
            }}
          >
            {expertStats.map((stat) => (
              <ExpertBucket
                key={stat.expertIdx}
                stat={stat}
                expertCapacity={expertCapacity}
                color={expertColors[stat.expertIdx % expertColors.length]}
              />
            ))}
          </Stack>

          {/* Legend */}
          <Stack direction="row" spacing={3} sx={{ mt: 4, justifyContent: 'center' }}>
            <LegendItem color="#4CAF50" label="Accepted Tokens" />
            <LegendItem color="#FF5252" label="Dropped Tokens" />
            <LegendItem
              color={theme.palette.divider}
              label="Empty Capacity (Wasted)"
            />
          </Stack>
        </Paper>

        {/* Insight Text */}
        <Paper elevation={1} sx={{ p: 3, backgroundColor: 'info.lighter' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Key Insight
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Capacity factor is a critical tradeoff:
            <br />
            • <strong>Too Low (0.5-0.8)</strong>: Many tokens get dropped, losing information and
            hurting model quality
            <br />
            • <strong>Too High (1.5-2.0)</strong>: Experts have empty slots, wasting memory and
            compute without benefit
            <br />
            • <strong>Sweet Spot (1.0-1.25)</strong>: The paper shows this balances quality and
            efficiency
          </Typography>
        </Paper>
      </Stack>
    </Box>
  );
};

/**
 * StatBox: Shows a single statistic
 */
const StatBox = ({ label, value, highlight = false }) => {
  const theme = useTheme();
  return (
    <Box
      sx={{
        flex: 1,
        textAlign: 'center',
        p: 2,
        borderRadius: 1,
        backgroundColor: highlight ? theme.palette.error.lighter : 'transparent',
        border: highlight ? `1px solid ${theme.palette.error.light}` : 'none',
      }}
    >
      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
        {label}
      </Typography>
      <Typography
        variant="h5"
        sx={{
          fontWeight: 'bold',
          color: highlight ? 'error.main' : 'primary.main',
          mt: 0.5,
        }}
      >
        {value}
      </Typography>
    </Box>
  );
};

/**
 * ExpertBucket: Visualizes a single expert's token allocation
 */
const ExpertBucket = ({ stat, expertCapacity, color }) => {
  const theme = useTheme();
  const bucketHeight = 280;
  const tokenSize = 28;
  const maxTokensPerColumn = Math.floor(bucketHeight / (tokenSize + 4));

  // Generate token positions for accepted tokens
  const acceptedTokenPositions = Array.from({ length: stat.acceptedTokens }).map(
    (_, idx) => ({
      id: `accepted-${idx}`,
      type: 'accepted',
      idx,
    })
  );

  // Generate token positions for dropped tokens
  const droppedTokenPositions = Array.from({ length: stat.droppedTokens }).map(
    (_, idx) => ({
      id: `dropped-${idx}`,
      type: 'dropped',
      idx: stat.acceptedTokens + idx,
    })
  );

  // Generate empty slots
  const emptySlots = Array.from({
    length: Math.max(0, expertCapacity - stat.acceptedTokens),
  }).map((_, idx) => ({
    id: `empty-${idx}`,
    idx: stat.acceptedTokens + idx,
  }));

  const capacityLine = (stat.acceptedTokens / expertCapacity) * bucketHeight;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {/* Bucket Container */}
      <Box
        sx={{
          position: 'relative',
          width: 120,
          height: bucketHeight,
          border: `2px solid ${color}`,
          borderRadius: 1,
          backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column-reverse',
          p: 1,
          gap: 1,
        }}
      >
        {/* Empty Slots (shown as dashed outlines) */}
        {emptySlots.map((slot) => (
          <motion.div
            key={slot.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.3 }}
            transition={{ duration: 0.3 }}
            style={{
              width: tokenSize,
              height: tokenSize,
              borderRadius: 4,
              border: `2px dashed ${theme.palette.divider}`,
              flexShrink: 0,
            }}
          />
        ))}

        {/* Accepted Tokens */}
        {acceptedTokenPositions.map((token) => (
          <motion.div
            key={token.id}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{
              duration: 0.4,
              delay: token.idx * 0.05,
              type: 'spring',
              stiffness: 200,
            }}
            style={{
              width: tokenSize,
              height: tokenSize,
              borderRadius: 4,
              backgroundColor: color,
              flexShrink: 0,
              boxShadow: `0 2px 8px rgba(0,0,0,0.15)`,
            }}
          />
        ))}

        {/* Capacity Line Indicator */}
        {stat.droppedTokens > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            style={{
              position: 'absolute',
              bottom: capacityLine,
              left: 0,
              right: 0,
              height: 2,
              backgroundColor: theme.palette.error.main,
              zIndex: 10,
            }}
          />
        )}
      </Box>

      {/* Dropped Tokens (shown below bucket, falling off) */}
      {droppedTokenPositions.length > 0 && (
        <Box
          sx={{
            mt: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            alignItems: 'center',
          }}
        >
          {droppedTokenPositions.slice(0, 2).map((token) => (
            <motion.div
              key={token.id}
              initial={{ y: 0, opacity: 1, scale: 1 }}
              animate={{ y: 20, opacity: 0.4, scale: 0.8 }}
              transition={{
                duration: 1.5,
                delay: token.idx * 0.1,
                repeat: Infinity,
                repeatType: 'mirror',
              }}
              style={{
                width: tokenSize,
                height: tokenSize,
                borderRadius: 4,
                backgroundColor: theme.palette.error.light,
                border: `2px solid ${theme.palette.error.main}`,
                opacity: 0.6,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  fontSize: '0.6rem',
                  fontWeight: 'bold',
                  color: 'error.main',
                }}
              >
                X
              </Typography>
            </motion.div>
          ))}
          {droppedTokenPositions.length > 2 && (
            <Typography variant="caption" sx={{ color: 'error.main', fontWeight: 600 }}>
              +{droppedTokenPositions.length - 2} dropped
            </Typography>
          )}
        </Box>
      )}

      {/* Expert Label and Info */}
      <Box sx={{ mt: 2, textAlign: 'center', width: '100%' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
          Expert {stat.expertIdx}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {stat.acceptedTokens}/{expertCapacity} tokens
        </Typography>
        {stat.droppedTokens > 0 && (
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              color: 'error.main',
              fontWeight: 600,
              mt: 0.5,
            }}
          >
            {stat.droppedTokens} dropped
          </Typography>
        )}
      </Box>
    </Box>
  );
};

/**
 * LegendItem: Shows a legend entry
 */
const LegendItem = ({ color, label }) => (
  <Stack direction="row" spacing={1} alignItems="center">
    <Box
      sx={{
        width: 16,
        height: 16,
        borderRadius: '2px',
        backgroundColor: color,
        border: color === '#FF5252' ? `1px solid ${color}` : 'none',
      }}
    />
    <Typography variant="caption">{label}</Typography>
  </Stack>
);

export default ExpertCapacityDiagram;
