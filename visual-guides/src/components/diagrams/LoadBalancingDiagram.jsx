import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Slider,
  Typography,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import WarningIcon from '@mui/icons-material/Warning';

const LoadBalancingDiagram = () => {
  const theme = useTheme();
  const [alpha, setAlpha] = useState(0.01);

  // Expert colors from theme
  const expertColors = theme.palette.expert?.colors || [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
  ];

  // Initial imbalanced distribution
  const imbalancedDistribution = [0.40, 0.05, 0.45, 0.10, 0.00, 0.00, 0.00, 0.00];

  // Calculate balanced distribution based on alpha
  // As alpha increases, distribution moves toward uniform (12.5% each)
  const calculateBalancedDistribution = () => {
    const uniform = 1 / 8; // 12.5% for each expert
    const balanceFactor = Math.min(alpha / 0.05, 1); // Max balance at alpha=0.05

    return imbalancedDistribution.map((val) => {
      return val * (1 - balanceFactor) + uniform * balanceFactor;
    });
  };

  const balancedDistribution = useMemo(
    () => calculateBalancedDistribution(),
    [alpha]
  );

  // Calculate auxiliary loss
  const calculateLoss = (distribution) => {
    const uniform = 1 / 8;
    return distribution.reduce((sum, frac) => {
      // Simplified: loss = alpha * fraction * fraction (proxy for router probability)
      return sum + frac * frac;
    }, 0) * alpha;
  };

  const imbalancedLoss = calculateLoss(imbalancedDistribution);
  const balancedLoss = calculateLoss(balancedDistribution);

  const showQualityWarning = alpha > 0.05;

  // Bar chart component
  const BarChart = ({ distribution, title, loss }) => (
    <Paper
      elevation={2}
      sx={{
        p: 3,
        backgroundColor: theme.palette.background.paper,
        borderRadius: 2,
      }}
    >
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
        {title}
      </Typography>

      {/* Expert bars */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, mb: 2 }}>
        {distribution.map((value, idx) => (
          <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              sx={{
                width: 60,
                fontSize: '0.875rem',
                fontWeight: 500,
              }}
            >
              Expert {idx}
            </Typography>
            <Box
              sx={{
                flex: 1,
                height: 32,
                backgroundColor: theme.palette.grey[200],
                borderRadius: 1,
                overflow: 'hidden',
                position: 'relative',
              }}
            >
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${value * 100}%` }}
                transition={{ duration: 0.6, ease: 'easeInOut' }}
                style={{
                  height: '100%',
                  backgroundColor: expertColors[idx % expertColors.length],
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                  paddingRight: 8,
                }}
              >
                <Typography
                  sx={{
                    color: '#fff',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    textShadow: '0 1px 2px rgba(0,0,0,0.3)',
                  }}
                >
                  {(value * 100).toFixed(1)}%
                </Typography>
              </motion.div>
            </Box>
          </Box>
        ))}
      </Box>

      {/* Loss metric */}
      <Box
        sx={{
          p: 1.5,
          backgroundColor: theme.palette.grey[100],
          borderRadius: 1,
          border: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            color: theme.palette.text.secondary,
            mb: 0.5,
          }}
        >
          Auxiliary Loss (α × Σ(f_i²))
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontWeight: 700,
            fontFamily: 'monospace',
            color: theme.palette.primary.main,
          }}
        >
          {loss.toFixed(6)}
        </Typography>
      </Box>
    </Paper>
  );

  return (
    <Box
      sx={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
        p: 2,
      }}
    >
      {/* Title */}
      <Box sx={{ mb: 2 }}>
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            mb: 1,
          }}
        >
          Auxiliary Load Balancing Loss
        </Typography>
        <Typography
          variant="body2"
          sx={{
            color: theme.palette.text.secondary,
          }}
        >
          Interactive visualization of Switch Transformer load balancing
        </Typography>
      </Box>

      {/* Alpha Slider */}
      <Paper
        elevation={1}
        sx={{
          p: 3,
          backgroundColor: theme.palette.mode === 'dark'
            ? theme.palette.grey[900]
            : theme.palette.grey[50],
          borderRadius: 2,
        }}
      >
        <Typography
          variant="subtitle1"
          sx={{
            fontWeight: 600,
            mb: 2,
          }}
        >
          Balancing Strength (α)
        </Typography>

        <Box sx={{ px: 1, mb: 2 }}>
          <Slider
            value={alpha}
            onChange={(e, newValue) => setAlpha(newValue)}
            min={0}
            max={0.1}
            step={0.001}
            marks={[
              { value: 0, label: '0.0' },
              { value: 0.01, label: '0.01 (default)' },
              { value: 0.05, label: '0.05 (warning)' },
              { value: 0.1, label: '0.1' },
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => value.toFixed(4)}
            sx={{
              '& .MuiSlider-markLabel': {
                fontSize: '0.75rem',
              },
            }}
          />
        </Box>

        {/* Alpha Explanation */}
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            color: theme.palette.text.secondary,
            mt: 2,
          }}
        >
          α controls how strongly the auxiliary loss penalizes unbalanced expert utilization.
          Increase it to enforce more uniform token distribution across experts.
        </Typography>

        {/* Quality Warning */}
        {showQualityWarning && (
          <Box
            sx={{
              display: 'flex',
              gap: 1,
              mt: 2,
              p: 1.5,
              backgroundColor: theme.palette.warning.lighter || '#FFF3CD',
              border: `1px solid ${theme.palette.warning.light || '#FFE69C'}`,
              borderRadius: 1,
              alignItems: 'flex-start',
            }}
          >
            <WarningIcon
              sx={{
                fontSize: '1.25rem',
                color: theme.palette.warning.main,
                mt: 0.25,
                flexShrink: 0,
              }}
            />
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 600,
                  color: theme.palette.warning.dark,
                  mb: 0.5,
                }}
              >
                Quality Degradation Risk
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: theme.palette.warning.dark,
                }}
              >
                α &gt; 0.05 forces experts to become too similar. The auxiliary loss overwhelms
                the primary task loss, degrading model quality. Keep α small!
              </Typography>
            </Box>
          </Box>
        )}
      </Paper>

      {/* Problem and Solution Grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
          gap: 3,
        }}
      >
        {/* Before Balancing */}
        <BarChart
          distribution={imbalancedDistribution}
          title="The Problem (No Balancing)"
          loss={imbalancedLoss}
        />

        {/* After Balancing */}
        <BarChart
          distribution={balancedDistribution}
          title="After Auxiliary Loss (With Balancing)"
          loss={balancedLoss}
        />
      </Box>

      {/* Formula Explanation */}
      <Paper
        elevation={2}
        sx={{
          p: 3,
          backgroundColor: theme.palette.mode === 'dark'
            ? theme.palette.grey[900]
            : theme.palette.grey[50],
          borderRadius: 2,
          border: `2px solid ${theme.palette.primary.main}`,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            mb: 2,
            color: theme.palette.primary.main,
          }}
        >
          The Auxiliary Loss Formula
        </Typography>

        {/* Plain English Explanation */}
        <Box sx={{ mb: 2 }}>
          <Typography
            variant="body2"
            sx={{
              mb: 1,
              fontFamily: 'monospace',
              backgroundColor: theme.palette.background.paper,
              p: 1.5,
              borderRadius: 1,
              border: `1px solid ${theme.palette.divider}`,
            }}
          >
            Loss = α × Σ(fraction_of_tokens_to_expert × average_router_probability_for_expert)
          </Typography>

          <Typography
            variant="body2"
            sx={{
              color: theme.palette.text.secondary,
              mt: 1.5,
            }}
          >
            <strong>What this means:</strong> The auxiliary loss measures how concentrated
            token flow is across experts. It is minimized when tokens are evenly distributed.
            The loss penalizes the router for sending too many tokens to any single expert.
          </Typography>

          <Typography
            variant="body2"
            sx={{
              color: theme.palette.text.secondary,
              mt: 1,
            }}
          >
            <strong>The α coefficient:</strong> A small scaling factor (typically 0.01) that
            prevents the auxiliary loss from dominating the primary cross-entropy loss during
            training. If α is too large, the model sacrifices accuracy for load balancing.
          </Typography>
        </Box>
      </Paper>

      {/* Key Insights */}
      <Paper
        elevation={1}
        sx={{
          p: 3,
          backgroundColor: theme.palette.success.lighter || '#E8F5E9',
          border: `1px solid ${theme.palette.success.light || '#C8E6C9'}`,
          borderRadius: 2,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            mb: 2,
            color: theme.palette.success.dark,
          }}
        >
          Key Insights from Switch Transformer
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Box>
            <Typography
              variant="subtitle2"
              sx={{
                fontWeight: 600,
                color: theme.palette.success.dark,
                mb: 0.5,
              }}
            >
              ✓ Simplified Load Balancing
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: theme.palette.success.dark,
              }}
            >
              Switch Transformer uses a single differentiable loss, unlike Shazeer et al. (2017)
              which required complex external balancing logic.
            </Typography>
          </Box>

          <Box>
            <Typography
              variant="subtitle2"
              sx={{
                fontWeight: 600,
                color: theme.palette.success.dark,
                mb: 0.5,
              }}
            >
              ✓ Minimal Overhead
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: theme.palette.success.dark,
              }}
            >
              The α coefficient is kept very small (0.01) so balancing doesn't interfere with
              the model's ability to learn the primary task.
            </Typography>
          </Box>

          <Box>
            <Typography
              variant="subtitle2"
              sx={{
                fontWeight: 600,
                color: theme.palette.success.dark,
                mb: 0.5,
              }}
            >
              ✓ Automatic Expert Specialization
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: theme.palette.success.dark,
              }}
            >
              By gently encouraging balanced utilization, experts naturally specialize in
              different linguistic phenomena without explicit routing rules.
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default LoadBalancingDiagram;
