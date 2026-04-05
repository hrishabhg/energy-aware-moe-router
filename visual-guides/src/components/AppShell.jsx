import { useState } from 'react';
import {
  AppBar, Toolbar, Typography, IconButton, Tabs, Tab, Box, Chip,
  useTheme, Container,
} from '@mui/material';
import { DarkMode, LightMode, AutoStories } from '@mui/icons-material';
import { useThemeToggle } from '../theme/ThemeContext';
import SwitchTransformerPage from '../pages/papers/SwitchTransformerPage';

const papers = [
  {
    id: 'switch-transformer',
    label: 'P2: Switch Transformer',
    shortLabel: 'Switch Transformer',
    year: 2021,
    component: SwitchTransformerPage,
  },
  // Future papers will be added here as tabs
];

export default function AppShell() {
  const [activeTab, setActiveTab] = useState(0);
  const { mode, toggle } = useThemeToggle();
  const theme = useTheme();

  const ActiveComponent = papers[activeTab].component;

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          bgcolor: 'background.paper',
          borderBottom: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Toolbar sx={{ gap: 1.5 }}>
          <AutoStories sx={{ color: 'primary.main', fontSize: 28 }} />
          <Typography variant="h6" color="text.primary" sx={{ flexGrow: 0, mr: 2 }}>
            Visual Guides
          </Typography>
          <Chip
            label="Energy-Aware MoE"
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
          <Box sx={{ flexGrow: 1 }} />
          <IconButton onClick={toggle} color="inherit" sx={{ color: 'text.secondary' }}>
            {mode === 'light' ? <DarkMode /> : <LightMode />}
          </IconButton>
        </Toolbar>
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            px: 2,
            '& .MuiTab-root': {
              fontSize: '0.875rem',
            },
          }}
        >
          {papers.map((p) => (
            <Tab
              key={p.id}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {p.label}
                  <Chip label={p.year} size="small" variant="outlined" sx={{ height: 20, fontSize: '0.7rem' }} />
                </Box>
              }
            />
          ))}
        </Tabs>
      </AppBar>
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <ActiveComponent />
      </Container>
    </Box>
  );
}
