import { createContext, useContext, useState, useMemo } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { lightTheme, darkTheme } from './theme';

const ThemeContext = createContext();

export const useThemeToggle = () => useContext(ThemeContext);

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState('light');

  const toggle = () => setMode((prev) => (prev === 'light' ? 'dark' : 'light'));
  const theme = useMemo(() => (mode === 'light' ? lightTheme : darkTheme), [mode]);

  return (
    <ThemeContext.Provider value={{ mode, toggle }}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
}
