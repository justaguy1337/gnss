import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { DarkModeProvider } from './contexts/DarkModeContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DataUpload from './pages/DataUpload';
import ModelInsights from './pages/ModelInsights';
import MetaLearner from './pages/MetaLearner';
import EarthVisualization from './pages/EarthVisualization';
import ImpactBenefits from './pages/ImpactBenefits';
import Research from './pages/Research';

function App() {
  return (
    <DarkModeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="/upload" element={<DataUpload />} />
            <Route path="/insights" element={<ModelInsights />} />
            <Route path="/meta-learner" element={<MetaLearner />} />
            <Route path="/earth" element={<EarthVisualization />} />
            <Route path="/impact" element={<ImpactBenefits />} />
            <Route path="/research" element={<Research />} />
          </Route>
        </Routes>
      </Router>
    </DarkModeProvider>
  );
}

export default App;
