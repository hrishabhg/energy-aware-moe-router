import React, { useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Paper,
  Typography,
  useTheme,
  Grid,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';

const SwitchRoutingDiagram = () => {
  const theme = useTheme();

  // Sample tokens
  const tokens = ['The', 'model', 'scales', 'efficiently'];

  // Sample experts count
  const numExperts = 4;
  const expertLabels = Array.from({ length: numExperts }, (_, i) => `Expert ${i}`);

  // Simulate routing: each token goes to ONE expert (top-1)
  // For demo: deterministic but varied assignment
  const routing = tokens.map((token, idx) => ({
    token,
    expertIndex: idx % numExperts,
    probability: (0.75 + Math.random() * 0.20).toFixed(2), // p between 0.75-0.95
  }));

  const [isAnimating, setIsAnimating] = useState(false);
  const [showRouting, setShowRouting] = useState(false);

  const handleRouteTokens = () => {
    setShowRouting(false);
    setIsAnimating(true);
    setTimeout(() => {
      setShowRouting(true);
      setIsAnimating(false);
    }, 1500);
  };

  // Get expert colors from theme
  const expertColors = theme.palette.expert?.colors || [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.error.main,
    theme.palette.info.main,
    '#FF6B9D',
    '#C44569',
  ];

  return (
    <Box sx={{ p: 3, bgcolor: theme.palette.background.paper }}>
      <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
        Switch Transformer: Top-1 Token Routing
      </Typography>

      <Grid container spacing={4}>
        {/* Main Diagram */}
        <Grid item xs={12} md={8}>
          <Paper
            sx={{
              p: 4,
              bgcolor: theme.palette.background.default,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 2,
              minHeight: 500,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
            }}
          >
            {/* Input Tokens */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" sx={{ display: 'block', mb: 1, fontWeight: 600 }}>
                Input Tokens
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {tokens.map((token, idx) => (
                  <motion.div
                    key={`token-${idx}`}
                    animate={
                      isAnimating
                        ? {
                            y: 300,
                            opacity: 0,
                          }
                        : { y: 0, opacity: 1 }
                    }
                    transition={{ duration: 0.8, ease: 'easeInOut' }}
                  >
                    <Chip
                      label={token}
                      sx={{
                        bgcolor: expertColors[routing[idx].expertIndex % expertColors.length],
                        color: '#fff',
                        fontWeight: 600,
                      }}
                    />
                  </motion.div>
                ))}
              </Box>
            </Box>

            {/* Router Box */}
            <Box sx={{ mb: 2, textAlign: 'center' }}>
              <motion.div
                animate={{
                  scale: isAnimating ? 1.05 : 1,
                }}
                transition={{ duration: 0.8 }}
              >
                <Paper
                  sx={{
                    p: 2,
                    bgcolor: theme.palette.action.hover,
                    border: `2px solid ${theme.palette.primary.main}`,
                    borderRadius: 1,
                    display: 'inline-block',
                    minWidth: 150,
                  }}
                >
                  <Typography sx={{ fontWeight: 600, fontSize: '0.95rem' }}>
                    Router
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                    argmax(logits) → 1 expert
                  </Typography>
                </Paper>
              </motion.div>
            </Box>

            {/* Expert Row */}
            <Box>
              <Typography variant="caption" sx={{ display: 'block', mb: 1, fontWeight: 600 }}>
                Experts
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'space-around' }}>
                {expertLabels.map((label, idx) => {
                  // Count tokens routed to this expert
                  const tokenCount = routing.filter(r => r.expertIndex === idx).length;

                  return (
                    <Box key={`expert-${idx}`} sx={{ textAlign: 'center' }}>
                      <motion.div
                        animate={{
                          scale: showRouting && tokenCount > 0 ? 1.1 : 1,
                        }}
                        transition={{ duration: 0.4 }}
                      >
                        <Paper
                          sx={{
                            p: 2,
                            bgcolor: expertColors[idx % expertColors.length],
                            color: '#fff',
                            minWidth: 80,
                            minHeight: 80,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center',
                            alignItems: 'center',
                            borderRadius: 1,
                            opacity: 0.85,
                            '&:hover': {
                              opacity: 1,
                            },
                            transition: 'opacity 0.2s',
                          }}
                        >
                          <Typography sx={{ fontWeight: 600, fontSize: '0.9rem' }}>
                            {label}
                          </Typography>
                          {showRouting && (
                            <motion.div
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ delay: 0.2, duration: 0.3 }}
                            >
                              <Typography variant="h6" sx={{ mt: 0.5 }}>
                                {tokenCount}
                              </Typography>
                            </motion.div>
                          )}
                        </Paper>
                      </motion.div>
                    </Box>
                  );
                })}
              </Box>
            </Box>

            {/* Routing Details */}
            <AnimatePresence>
              {showRouting && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  transition={{ duration: 0.4 }}
                  style={{ marginTop: 24 }}
                >
                  <Paper
                    sx={{
                      p: 2,
                      bgcolor: theme.palette.action.hover,
                      borderRadius: 1,
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                      Routing Assignment
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                      {routing.map((r, idx) => (
                        <Box
                          key={`routing-${idx}`}
                          sx={{
                            px: 1.5,
                            py: 0.5,
                            bgcolor: theme.palette.background.paper,
                            borderRadius: 1,
                            border: `1px solid ${theme.palette.divider}`,
                          }}
                        >
                          <Typography variant="caption">
                            <span style={{ fontWeight: 600, color: expertColors[r.expertIndex % expertColors.length] }}>
                              {r.token}
                            </span>
                            {' → '}
                            <span style={{ fontWeight: 600 }}>{expertLabels[r.expertIndex]}</span>
                            {' '}
                            <span style={{ opacity: 0.7 }}>(p={r.probability})</span>
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                </motion.div>
              )}
            </AnimatePresence>
          </Paper>
        </Grid>

        {/* Right Panel: Comparison & Button */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Route Button */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                variant="contained"
                color="primary"
                onClick={handleRouteTokens}
                disabled={isAnimating}
                fullWidth
                sx={{ py: 1.5, fontWeight: 600 }}
              >
                {isAnimating ? 'Routing...' : 'Route Tokens'}
              </Button>
            </motion.div>

            {/* Comparison Panel */}
            <Paper
              sx={{
                p: 2.5,
                bgcolor: theme.palette.background.default,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: 2,
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                Routing Comparison
              </Typography>

              {/* Traditional MoE */}
              <Box sx={{ mb: 2.5 }}>
                <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, mb: 1 }}>
                  Traditional MoE (top-2)
                </Typography>
                <Box
                  sx={{
                    p: 1.5,
                    bgcolor: theme.palette.action.hover,
                    borderRadius: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                  }}
                >
                  {/* SVG Icon for top-2 routing */}
                  <svg
                    width="80"
                    height="60"
                    viewBox="0 0 80 60"
                    style={{ flexShrink: 0 }}
                  >
                    {/* Input token */}
                    <circle cx="10" cy="30" r="4" fill={theme.palette.primary.main} />

                    {/* Two output experts */}
                    <circle cx="70" cy="15" r="4" fill={theme.palette.secondary.main} />
                    <circle cx="70" cy="45" r="4" fill={theme.palette.warning.main} />

                    {/* Lines: token to both experts */}
                    <line x1="14" y1="26" x2="66" y2="15" stroke={theme.palette.secondary.main} strokeWidth="1.5" strokeDasharray="3,3" opacity="0.6" />
                    <line x1="14" y1="34" x2="66" y2="45" stroke={theme.palette.warning.main} strokeWidth="1.5" strokeDasharray="3,3" opacity="0.6" />
                  </svg>

                  <Box>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                      Each token routes to <strong>2 experts</strong>
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mt: 0.5 }}>
                      Higher communication overhead
                    </Typography>
                  </Box>
                </Box>
              </Box>

              {/* Switch Transformer */}
              <Box>
                <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, mb: 1 }}>
                  Switch Transformer (top-1)
                </Typography>
                <Box
                  sx={{
                    p: 1.5,
                    bgcolor: theme.palette.success.lighter || theme.palette.action.hover,
                    borderRadius: 1,
                    border: `1px solid ${theme.palette.success.main}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                  }}
                >
                  {/* SVG Icon for top-1 routing */}
                  <svg
                    width="80"
                    height="60"
                    viewBox="0 0 80 60"
                    style={{ flexShrink: 0 }}
                  >
                    {/* Input token */}
                    <circle cx="10" cy="30" r="4" fill={theme.palette.success.main} />

                    {/* One output expert (simplified) */}
                    <circle cx="70" cy="30" r="4" fill={theme.palette.success.main} />

                    {/* Single line: token to one expert */}
                    <line x1="14" y1="30" x2="66" y2="30" stroke={theme.palette.success.main} strokeWidth="2" opacity="0.8" />
                  </svg>

                  <Box>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                      Each token routes to <strong>1 expert</strong>
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mt: 0.5 }}>
                      Minimal overhead, simplified gradients
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Paper>

            {/* Key Insights */}
            <Paper
              sx={{
                p: 2,
                bgcolor: theme.palette.info.lighter || theme.palette.action.hover,
                border: `1px solid ${theme.palette.info.main}`,
                borderRadius: 2,
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                Key Benefits
              </Typography>
              <Box component="ul" sx={{ pl: 2, m: 0, py: 0.5 }}>
                <Typography component="li" variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                  Deterministic routing via argmax
                </Typography>
                <Typography component="li" variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                  Reduced communication
                </Typography>
                <Typography component="li" variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                  Simpler gradient flow
                </Typography>
                <Typography component="li" variant="caption">
                  Potential load imbalance
                </Typography>
              </Box>
            </Paper>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SwitchRoutingDiagram;
