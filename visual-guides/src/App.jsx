import { ThemeProvider } from './theme/ThemeContext';
import AppShell from './components/AppShell';

export default function App() {
  return (
    <ThemeProvider>
      <AppShell />
    </ThemeProvider>
  );
}
