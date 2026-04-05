import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  useTheme,
  IconButton,
  Tooltip,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';

const ArchitectureComparisonDiagram = () => {
  const theme = useTheme();
  const [hoveredLayer, setHoveredLayer] = useState(null);
  const [showDataFlow, setShowDataFlow] = useState(false);
  const [activeColumn, setActiveColumn] = useState(null);

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

  const blockHeight = 60;
  const blockWidth = 100;
  const spacing = 80;
  const columnSpacing = 300;

  // Dense Transformer layers
  const denseLayers = [
    { id: 'dense-embed', label: 'Input', type: 'embedding' },
    { id: 'dense-attn1', label: 'Self-Attn', type: 'attention' },
    { id: 'dense-ffn1', label: 'FFN', type: 'ffn' },
    { id: 'dense-attn2', label: 'Self-Attn', type: 'attention' },
    { id: 'dense-ffn2', label: 'FFN', type: 'ffn' },
    { id: 'dense-out', label: 'Output', type: 'output' },
  ];

  // MoE Transformer layers
  const moeLayers = [
    { id: 'moe-embed', label: 'Input', type: 'embedding' },
    { id: 'moe-attn1', label: 'Self-Attn', type: 'attention' },
    { id: 'moe-moe1', label: 'MoE (top-2)', type: 'moe' },
    { id: 'moe-attn2', label: 'Self-Attn', type: 'attention' },
    { id: 'moe-moe2', label: 'MoE (top-2)', type: 'moe' },
    { id: 'moe-out', label: 'Output', type: 'output' },
  ];

  // Switch Transformer layers
  const switchLayers = [
    { id: 'switch-embed', label: 'Input', type: 'embedding' },
    { id: 'switch-attn1', label: 'Self-Attn', type: 'attention' },
    { id: 'switch-switch1', label: 'Switch (top-1)', type: 'switch' },
    { id: 'switch-attn2', label: 'Self-Attn', type: 'attention' },
    { id: 'switch-switch2', label: 'Switch (top-1)', type: 'switch' },
    { id: 'switch-out', label: 'Output', type: 'output' },
  ];

  // Layer block component
  const LayerBlock = ({ layer, yPos, columnIndex, isHovered, onHover }) => {
    const getColor = () => {
      if (layer.type === 'embedding' || layer.type === 'output') {
        return theme.palette.grey[400];
      }
      if (layer.type === 'attention') {
        return theme.palette.primary.main;
      }
      if (layer.type === 'ffn') {
        return expertColors[0];
      }
      if (layer.type === 'moe' || layer.type === 'switch') {
        return theme.palette.secondary.main;
      }
      return theme.palette.grey[300];
    };

    return (
      <motion.g
        onMouseEnter={() => onHover(layer.id)}
        onMouseLeave={() => onHover(null)}
        animate={{
          opacity: !hoveredLayer || isHovered ? 1 : 0.4,
        }}
        transition={{ duration: 0.2 }}
      >
        <rect
          x={0}
          y={yPos}
          width={blockWidth}
          height={blockHeight}
          fill={getColor()}
          stroke={theme.palette.text.primary}
          strokeWidth={isHovered ? 2 : 1}
          rx={4}
        />
        <text
          x={blockWidth / 2}
          y={yPos + blockHeight / 2 + 6}
          textAnchor="middle"
          fontSize="12"
          fontWeight={isHovered ? 'bold' : 'normal'}
          fill={theme.palette.getContrastText(getColor())}
        >
          {layer.label}
        </text>
      </motion.g>
    );
  };

  // Expert boxes for MoE layer
  const MoEExpertVisualization = ({ yPos, columnX }) => {
    const routerX = blockWidth / 2;
    const expertSpacing = 60;
    const expertSize = 40;

    return (
      <g>
        {/* Router box */}
        <rect
          x={0}
          y={yPos}
          width={blockWidth}
          height={blockHeight}
          fill={theme.palette.secondary.main}
          stroke={theme.palette.text.primary}
          strokeWidth={1}
          rx={4}
        />
        <text
          x={blockWidth / 2}
          y={yPos + blockHeight / 2 + 6}
          textAnchor="middle"
          fontSize="11"
          fill={theme.palette.getContrastText(theme.palette.secondary.main)}
          fontWeight="bold"
        >
          Router
        </text>

        {/* Expert boxes (top-2) */}
        <motion.g
          animate={{
            opacity: hoveredLayer === `moe-layer-${columnX}` ? 1 : 0.7,
          }}
          transition={{ duration: 0.2 }}
        >
          {/* Left expert */}
          <line
            x1={routerX}
            y1={yPos + blockHeight}
            x2={-expertSpacing / 2 + expertSize / 2}
            y2={yPos + blockHeight + 30}
            stroke={expertColors[1]}
            strokeWidth={2}
            markerEnd="url(#arrowhead)"
          />
          <rect
            x={-expertSpacing / 2}
            y={yPos + blockHeight + 30}
            width={expertSize}
            height={expertSize}
            fill={expertColors[1]}
            stroke={theme.palette.text.primary}
            strokeWidth={1}
            rx={3}
            opacity={0.8}
          />

          {/* Right expert */}
          <line
            x1={routerX}
            y1={yPos + blockHeight}
            x2={expertSpacing / 2 + expertSize / 2}
            y2={yPos + blockHeight + 30}
            stroke={expertColors[2]}
            strokeWidth={2}
            markerEnd="url(#arrowhead)"
          />
          <rect
            x={expertSpacing / 2}
            y={yPos + blockHeight + 30}
            width={expertSize}
            height={expertSize}
            fill={expertColors[2]}
            stroke={theme.palette.text.primary}
            strokeWidth={1}
            rx={3}
            opacity={0.8}
          />

          {/* Combining lines */}
          <line
            x1={-expertSpacing / 2 + expertSize / 2}
            y1={yPos + blockHeight + 30 + expertSize}
            x2={-15}
            y2={yPos + blockHeight + 70}
            stroke={expertColors[1]}
            strokeWidth={1.5}
            strokeDasharray="4,4"
            opacity={0.6}
          />
          <line
            x1={expertSpacing / 2 + expertSize / 2}
            y1={yPos + blockHeight + 30 + expertSize}
            x2={15}
            y2={yPos + blockHeight + 70}
            stroke={expertColors[2]}
            strokeWidth={1.5}
            strokeDasharray="4,4"
            opacity={0.6}
          />
        </motion.g>
      </g>
    );
  };

  // Expert box for Switch layer
  const SwitchExpertVisualization = ({ yPos }) => {
    const routerX = blockWidth / 2;
    const expertSize = 40;

    return (
      <g>
        {/* Router box */}
        <rect
          x={0}
          y={yPos}
          width={blockWidth}
          height={blockHeight}
          fill={theme.palette.secondary.main}
          stroke={theme.palette.text.primary}
          strokeWidth={1}
          rx={4}
        />
        <text
          x={blockWidth / 2}
          y={yPos + blockHeight / 2 + 6}
          textAnchor="middle"
          fontSize="11"
          fill={theme.palette.getContrastText(theme.palette.secondary.main)}
          fontWeight="bold"
        >
          Switch
        </text>

        {/* Single expert box */}
        <motion.g
          animate={{
            opacity: hoveredLayer === `switch-layer` ? 1 : 0.7,
          }}
          transition={{ duration: 0.2 }}
        >
          <line
            x1={routerX}
            y1={yPos + blockHeight}
            x2={routerX}
            y2={yPos + blockHeight + 30}
            stroke={expertColors[3]}
            strokeWidth={3}
            markerEnd="url(#arrowhead)"
          />
          <rect
            x={routerX - expertSize / 2}
            y={yPos + blockHeight + 30}
            width={expertSize}
            height={expertSize}
            fill={expertColors[3]}
            stroke={theme.palette.text.primary}
            strokeWidth={2}
            rx={3}
          />
          <text
            x={routerX}
            y={yPos + blockHeight + 30 + expertSize / 2 + 5}
            textAnchor="middle"
            fontSize="10"
            fill={theme.palette.getContrastText(expertColors[3])}
            fontWeight="bold"
          >
            Expert
          </text>
        </motion.g>
      </g>
    );
  };

  // Data flow visualization
  const DataFlowDot = ({ columnIndex, layers }) => {
    if (!showDataFlow) return null;

    const totalHeight = layers.length * spacing + blockHeight * layers.length;

    if (columnIndex === 0) {
      // Dense: straight path
      return (
        <motion.circle
          cx={blockWidth / 2}
          cy={20}
          r={4}
          fill={theme.palette.success.main}
          animate={{
            y: totalHeight - blockHeight,
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: 'linear',
          }}
        />
      );
    } else if (columnIndex === 1) {
      // MoE: splits and recombines
      return (
        <>
          {/* Path through left expert */}
          <motion.circle
            cx={blockWidth / 2 - 30}
            cy={20}
            r={4}
            fill={expertColors[1]}
            animate={{
              x: [-30, -30, -30, -30, -30, -30],
              y: [20, 140, 170, 170, 280, 350],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: 'linear',
              times: [0, 0.2, 0.3, 0.4, 0.6, 1],
            }}
          />
          {/* Path through right expert */}
          <motion.circle
            cx={blockWidth / 2 + 30}
            cy={20}
            r={4}
            fill={expertColors[2]}
            animate={{
              x: [30, 30, 30, 30, 30, 30],
              y: [20, 140, 170, 170, 280, 350],
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: 'linear',
              times: [0, 0.2, 0.3, 0.4, 0.6, 1],
            }}
          />
        </>
      );
    } else if (columnIndex === 2) {
      // Switch: single path
      return (
        <motion.circle
          cx={blockWidth / 2}
          cy={20}
          r={4}
          fill={expertColors[3]}
          animate={{
            y: totalHeight - blockHeight,
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: 'linear',
          }}
        />
      );
    }
  };

  // Column renderer
  const ColumnDiagram = ({ title, annotation, layers, columnIndex, params }) => {
    const totalHeight = layers.length * spacing + blockHeight * layers.length;

    return (
      <motion.div
        onMouseEnter={() => setActiveColumn(columnIndex)}
        onMouseLeave={() => setActiveColumn(null)}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: columnIndex * 0.1 }}
      >
        <Paper
          elevation={activeColumn === columnIndex ? 8 : 2}
          sx={{
            p: 3,
            background: theme.palette.background.paper,
            transition: 'all 0.3s ease',
            borderLeft: `4px solid ${expertColors[columnIndex % 8]}`,
          }}
        >
          <Typography
            variant="h6"
            sx={{
              mb: 1,
              fontWeight: 'bold',
              color: expertColors[columnIndex % 8],
            }}
          >
            {title}
          </Typography>

          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mb: 2,
              fontStyle: 'italic',
              color: theme.palette.text.secondary,
            }}
          >
            {annotation}
          </Typography>

          <Box sx={{ position: 'relative', mb: 2 }}>
            <svg
              width={blockWidth + 100}
              height={totalHeight + 100}
              style={{
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: '4px',
                background: theme.palette.mode === 'dark' ? '#1a1a1a' : '#fafafa',
              }}
            >
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="10"
                  refX="5"
                  refY="5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 5, 0 10" fill={theme.palette.text.primary} />
                </marker>
              </defs>

              <g transform={`translate(50, 30)`}>
                {/* Draw connecting lines */}
                {layers.map((layer, idx) => {
                  if (idx < layers.length - 1) {
                    return (
                      <line
                        key={`line-${idx}`}
                        x1={blockWidth / 2}
                        y1={idx * spacing + (idx + 1) * blockHeight}
                        x2={blockWidth / 2}
                        y2={idx * spacing + (idx + 1) * blockHeight + spacing - blockHeight}
                        stroke={theme.palette.divider}
                        strokeWidth={1}
                        markerEnd="url(#arrowhead)"
                      />
                    );
                  }
                  return null;
                })}

                {/* Draw layers */}
                {layers.map((layer, idx) => {
                  const yPos = idx * spacing + idx * blockHeight;

                  if (layer.type === 'moe') {
                    return (
                      <g key={layer.id} transform={`translate(0, ${yPos})`}>
                        <MoEExpertVisualization yPos={0} columnX={columnIndex} />
                      </g>
                    );
                  }

                  if (layer.type === 'switch') {
                    return (
                      <g key={layer.id} transform={`translate(0, ${yPos})`}>
                        <SwitchExpertVisualization yPos={0} />
                      </g>
                    );
                  }

                  return (
                    <LayerBlock
                      key={layer.id}
                      layer={layer}
                      yPos={yPos}
                      columnIndex={columnIndex}
                      isHovered={hoveredLayer === layer.id}
                      onHover={setHoveredLayer}
                    />
                  );
                })}

                {/* Data flow visualization */}
                <DataFlowDot columnIndex={columnIndex} layers={layers} />
              </g>
            </svg>
          </Box>

          <Typography
            variant="caption"
            sx={{
              display: 'block',
              color: theme.palette.text.secondary,
              fontWeight: 'bold',
            }}
          >
            {params}
          </Typography>
        </Paper>
      </motion.div>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Controls */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
          pb: 2,
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          Transformer Architecture Comparison
        </Typography>

        <Tooltip
          title={
            showDataFlow ? 'Hide token data flow' : 'Show token data flow visualization'
          }
        >
          <IconButton
            onClick={() => setShowDataFlow(!showDataFlow)}
            color={showDataFlow ? 'primary' : 'default'}
            size="large"
          >
            {showDataFlow ? <VisibilityIcon /> : <VisibilityOffIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      {/* Three-column layout */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 3,
        }}
      >
        <ColumnDiagram
          title="Dense Transformer"
          annotation="Every token goes through the SAME FFN"
          layers={denseLayers}
          columnIndex={0}
          params="Parameters: N"
        />

        <ColumnDiagram
          title="MoE Transformer (top-2)"
          annotation="Each token routed to TOP-2 experts, outputs combined"
          layers={moeLayers}
          columnIndex={1}
          params="Parameters: ~K×N (compute: ~2×)"
        />

        <ColumnDiagram
          title="Switch Transformer (top-1)"
          annotation="Each token routed to exactly ONE expert"
          layers={switchLayers}
          columnIndex={2}
          params="Parameters: ~K×N (compute: ~1×)"
        />
      </Box>

      {/* Legend */}
      <Paper
        elevation={0}
        sx={{
          mt: 4,
          p: 2,
          background: theme.palette.background.default,
          borderRadius: 1,
        }}
      >
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 'bold' }}>
          Interactive Elements:
        </Typography>
        <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
          • Hover over any layer to highlight it across all architectures
        </Typography>
        <Typography variant="caption" sx={{ display: 'block' }}>
          • Toggle data flow visibility to see how tokens travel through each
          architecture
        </Typography>
      </Paper>
    </Box>
  );
};

export default ArchitectureComparisonDiagram;
