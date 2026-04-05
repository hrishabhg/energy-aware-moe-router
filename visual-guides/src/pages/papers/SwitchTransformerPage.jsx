import { useState } from 'react';
import {
  Box, Typography, Tabs, Tab, Chip, Paper, Divider, useTheme, Alert,
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AccountTree, TuneRounded, Balance, CompareArrows, Speed, BugReport,
} from '@mui/icons-material';
import SwitchRoutingDiagram from '../../components/diagrams/SwitchRoutingDiagram';
import ExpertCapacityDiagram from '../../components/diagrams/ExpertCapacityDiagram';
import LoadBalancingDiagram from '../../components/diagrams/LoadBalancingDiagram';
import ArchitectureComparisonDiagram from '../../components/diagrams/ArchitectureComparisonDiagram';
import ScalingParallelismDiagram from '../../components/diagrams/ScalingParallelismDiagram';
import TrainingStabilityDiagram from '../../components/diagrams/TrainingStabilityDiagram';

const sections = [
  {
    id: 'architecture',
    label: 'Architecture',
    icon: <CompareArrows fontSize="small" />,
    component: ArchitectureComparisonDiagram,
    description: 'How Switch Transformer differs from Dense and traditional MoE architectures.',
  },
  {
    id: 'routing',
    label: 'Switch Routing',
    icon: <AccountTree fontSize="small" />,
    component: SwitchRoutingDiagram,
    description: 'The core innovation: routing each token to exactly one expert instead of top-k.',
  },
  {
    id: 'capacity',
    label: 'Expert Capacity',
    icon: <TuneRounded fontSize="small" />,
    component: ExpertCapacityDiagram,
    description: 'How the capacity factor controls token overflow and compute efficiency.',
  },
  {
    id: 'balancing',
    label: 'Load Balancing',
    icon: <Balance fontSize="small" />,
    component: LoadBalancingDiagram,
    description: 'The auxiliary loss that keeps experts evenly utilized.',
  },
  {
    id: 'scaling',
    label: 'Scaling & Parallelism',
    icon: <Speed fontSize="small" />,
    component: ScalingParallelismDiagram,
    description: 'How Switch Transformers scale to trillions of parameters across devices.',
  },
  {
    id: 'stability',
    label: 'Training Stability',
    icon: <BugReport fontSize="small" />,
    component: TrainingStabilityDiagram,
    description: 'Instability problems in sparse models and the three solutions proposed.',
  },
];

const fadeVariant = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -12 },
};

export default function SwitchTransformerPage() {
  const [activeSection, setActiveSection] = useState(0);
  const theme = useTheme();

  const ActiveDiagram = sections[activeSection].component;

  return (
    <Box>
      {/* Paper Header */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 3,
          border: `1px solid ${theme.palette.divider}`,
          background: theme.palette.mode === 'light'
            ? 'linear-gradient(135deg, #f8f6ff 0%, #f0fdfa 100%)'
            : 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, flexWrap: 'wrap' }}>
          <Box sx={{ flex: 1, minWidth: 300 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Chip label="Paper 2" size="small" color="primary" />
              <Chip label="Google Brain, 2021" size="small" variant="outlined" />
              <Chip label="arXiv:2101.03961" size="small" variant="outlined" />
            </Box>
            <Typography variant="h4" gutterBottom>
              Switch Transformers
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" gutterBottom>
              Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, maxWidth: 700 }}>
              Fedus, Zoph & Shazeer propose simplifying MoE routing from top-k to top-1,
              enabling models with up to 1.6 trillion parameters that train at the same
              compute cost as their dense counterparts while being significantly faster to converge.
            </Typography>
          </Box>
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              minWidth: 200,
              bgcolor: theme.palette.mode === 'light'
                ? 'rgba(108, 92, 231, 0.04)'
                : 'rgba(162, 155, 254, 0.06)',
            }}
          >
            <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
              Key Numbers
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              1.6T parameters
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Largest model trained
            </Typography>
            <Divider sx={{ my: 1 }} />
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              7× speedup
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Pre-training speed vs T5-Base
            </Typography>
            <Divider sx={{ my: 1 }} />
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Top-1 routing
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Core simplification over MoE
            </Typography>
          </Paper>
        </Box>
      </Paper>

      {/* Section Navigation */}
      <Paper
        elevation={0}
        sx={{
          mb: 3,
          border: `1px solid ${theme.palette.divider}`,
          overflow: 'hidden',
        }}
      >
        <Tabs
          value={activeSection}
          onChange={(_, v) => setActiveSection(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            bgcolor: theme.palette.mode === 'light'
              ? 'rgba(0,0,0,0.02)'
              : 'rgba(255,255,255,0.02)',
            '& .MuiTab-root': {
              minHeight: 56,
            },
          }}
        >
          {sections.map((s) => (
            <Tab
              key={s.id}
              icon={s.icon}
              iconPosition="start"
              label={s.label}
              sx={{ gap: 0.5 }}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Section Description */}
      <Alert
        severity="info"
        variant="outlined"
        sx={{ mb: 3, borderRadius: 2 }}
        icon={sections[activeSection].icon}
      >
        {sections[activeSection].description}
      </Alert>

      {/* Active Diagram */}
      <AnimatePresence mode="wait">
        <motion.div
          key={sections[activeSection].id}
          variants={fadeVariant}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{ duration: 0.25 }}
        >
          <ActiveDiagram />
        </motion.div>
      </AnimatePresence>
    </Box>
  );
}
