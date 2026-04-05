import { createTheme } from '@mui/material/styles';

const shared = {
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: { fontWeight: 700, letterSpacing: '-0.02em' },
    h4: { fontWeight: 700, letterSpacing: '-0.01em' },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    subtitle1: { fontWeight: 500, fontSize: '1.05rem' },
    body2: { lineHeight: 1.7 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none', fontWeight: 600, borderRadius: 8 },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: 'none' },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: { textTransform: 'none', fontWeight: 500, minHeight: 48 },
      },
    },
  },
};

export const lightTheme = createTheme({
  ...shared,
  palette: {
    mode: 'light',
    primary: { main: '#6C5CE7', light: '#A29BFE', dark: '#5A4BD1' },
    secondary: { main: '#00B894', light: '#55EFC4', dark: '#00A381' },
    background: {
      default: '#F8F9FC',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#2D3436',
      secondary: '#636E72',
    },
    divider: 'rgba(0,0,0,0.08)',
    expert: {
      colors: ['#6C5CE7', '#00B894', '#E17055', '#0984E3', '#FDCB6E', '#E84393', '#00CEC9', '#FD79A8'],
    },
  },
});

export const darkTheme = createTheme({
  ...shared,
  palette: {
    mode: 'dark',
    primary: { main: '#A29BFE', light: '#C4BFFF', dark: '#6C5CE7' },
    secondary: { main: '#55EFC4', light: '#81FDD8', dark: '#00B894' },
    background: {
      default: '#0F0F1A',
      paper: '#1A1A2E',
    },
    text: {
      primary: '#E8E8F0',
      secondary: '#A0A0B8',
    },
    divider: 'rgba(255,255,255,0.08)',
    expert: {
      colors: ['#A29BFE', '#55EFC4', '#FAB1A0', '#74B9FF', '#FFEAA7', '#FD79A8', '#81ECEC', '#FF7675'],
    },
  },
});
