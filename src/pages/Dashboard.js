import { useState, useEffect, useCallback } from 'react';
import { 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from 'recharts';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  Satellite,
  AlertTriangle,
  CheckCircle2,
  Wifi,
  WifiOff
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const Dashboard = () => {
  const [predictions, setPredictions] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [usingDemoData, setUsingDemoData] = useState(false);
  const [selectedHorizon, setSelectedHorizon] = useState(1);

  const horizons = [
    { value: 1, label: '15 min' },
    { value: 2, label: '30 min' },
    { value: 4, label: '1 hr' },
    { value: 8, label: '2 hr' },
    { value: 96, label: '24 hr' },
  ];

  // Generate realistic demo data matching the API shape
  const generateDemoData = useCallback(() => {
    const demoPred = {};
    const demoEval = {};

    horizons.forEach(({ value: h }) => {
      const n = 80;
      const preds = [], truths = [], uncertainties = [];
      for (let i = 0; i < n; i++) {
        const t = (i / n) * 2 * Math.PI;
        const truth = 2.5 * Math.sin(t) + 0.8 * Math.cos(2 * t) + (Math.random() - 0.5) * 0.3;
        const noise = (Math.random() - 0.5) * (0.08 + h * 0.005);
        preds.push(truth + noise);
        truths.push(truth);
        uncertainties.push(0.08 + h * 0.015 + Math.random() * 0.03);
      }
      const residuals = truths.map((t, i) => t - preds[i]);
      const rmse = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / n);
      const mae = residuals.reduce((s, r) => s + Math.abs(r), 0) / n;

      demoPred[h] = {
        predictions: preds, ground_truth: truths, uncertainties,
        residuals, horizon: h, horizon_min: h * 15, n_predictions: n,
        rmse, mae,
        base_predictions: {
          lstm_gru: preds.map(p => p + (Math.random() - 0.5) * 0.15),
          transformer: preds.map(p => p + (Math.random() - 0.5) * 0.12),
          xgboost: preds.map(p => p + (Math.random() - 0.5) * 0.20),
        }
      };

      demoEval[h] = {
        horizon: h, horizon_min: h * 15, rmse, mae,
        r2_score: 0.97 - h * 0.003,
        residual_mean: residuals.reduce((a, b) => a + b, 0) / n,
        residual_std: Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / n),
        shapiro_wilk: { p_value: 0.2 + Math.random() * 0.6, is_normal: true },
      };
    });

    return { predictions: demoPred, evaluation: demoEval };
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, predRes, evalRes, modelsRes] = await Promise.all([
          fetch(`${API_BASE}/status`).catch(() => null),
          fetch(`${API_BASE}/predict/all`).catch(() => null),
          fetch(`${API_BASE}/evaluation`).catch(() => null),
          fetch(`${API_BASE}/models`).catch(() => null),
        ]);

        if (statusRes?.ok) setApiStatus(await statusRes.json());
        if (predRes?.ok && evalRes?.ok) {
          setPredictions(await predRes.json());
          setEvaluation(await evalRes.json());
          if (modelsRes?.ok) setModelsInfo(await modelsRes.json());
          setUsingDemoData(false);
        } else {
          throw new Error('API not available');
        }
      } catch {
        const demo = generateDemoData();
        setPredictions(demo.predictions);
        setEvaluation(demo.evaluation);
        setUsingDemoData(true);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [generateDemoData]);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={48} style={{ color: 'var(--primary-500)', animation: 'pulse 2s infinite' }} />
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  // Compute summary metrics from evaluation data
  const bestHorizon = evaluation ? Object.values(evaluation).reduce((best, ev) =>
    (ev.rmse < best.rmse ? ev : best), { rmse: Infinity }) : null;
  const avgR2 = evaluation ? (Object.values(evaluation).reduce((s, ev) => s + (ev.r2_score || 0), 0) / Object.keys(evaluation).length) : 0;
  const allNormal = evaluation ? Object.values(evaluation).every(ev => ev.shapiro_wilk?.is_normal) : false;
  const nSatellites = apiStatus?.has_data ? 6 : 6; // from dataset

  const metrics = [
    {
      title: 'Satellites Tracked',
      value: `${nSatellites}`,
      change: '4 MEO + 2 GEO',
      changeType: 'positive',
      icon: Satellite,
    },
    {
      title: 'Best RMSE',
      value: bestHorizon ? bestHorizon.rmse.toFixed(4) : '—',
      change: bestHorizon ? `@ ${bestHorizon.horizon_min}min horizon` : '',
      changeType: 'positive',
      icon: TrendingUp,
    },
    {
      title: 'Avg R² Score',
      value: avgR2 ? (avgR2 * 100).toFixed(1) + '%' : '—',
      change: 'across all horizons',
      changeType: avgR2 > 0.9 ? 'positive' : 'negative',
      icon: Activity,
    },
    {
      title: 'Residual Normality',
      value: allNormal ? 'Normal ✓' : 'Mixed',
      change: allNormal ? 'Shapiro-Wilk p > 0.05' : 'Check per-horizon',
      changeType: allNormal ? 'positive' : 'negative',
      icon: allNormal ? CheckCircle2 : AlertTriangle,
    }
  ];

  // Build time-series data for selected horizon
  const currentPred = predictions?.[selectedHorizon];
  const timeSeriesData = currentPred ? currentPred.predictions.slice(0, 48).map((pred, i) => ({
    time: `${String(Math.floor(i * 15 / 60)).padStart(2, '0')}:${String((i * 15) % 60).padStart(2, '0')}`,
    predicted: parseFloat(pred.toFixed(3)),
    actual: currentPred.ground_truth ? parseFloat(currentPred.ground_truth[i]?.toFixed(3)) : null,
    upper: parseFloat((pred + (currentPred.uncertainties?.[i] || 0) * 2).toFixed(3)),
    lower: parseFloat((pred - (currentPred.uncertainties?.[i] || 0) * 2).toFixed(3)),
    uncertainty: parseFloat(((currentPred.uncertainties?.[i] || 0) * 2).toFixed(4)),
  })) : [];

  // Horizon comparison bar data
  const horizonCompData = horizons.map(({ value: h, label }) => {
    const ev = evaluation?.[h];
    return {
      horizon: label,
      rmse: ev ? parseFloat(ev.rmse.toFixed(5)) : 0,
      mae: ev ? parseFloat(ev.mae.toFixed(5)) : 0,
    };
  });

  // Ridge weight data for stacker visualization
  const ridgeWeightData = modelsInfo ? horizons.map(({ value: h, label }) => {
    const w = modelsInfo.horizons?.[h]?.weights || {};
    return {
      horizon: label,
      'LSTM-GRU': parseFloat((w['LSTM-GRU'] || 0.33).toFixed(3)),
      'Transformer': parseFloat((w['Transformer'] || 0.33).toFixed(3)),
      'XGBoost': parseFloat((w['XGBoost'] || 0.33).toFixed(3)),
    };
  }) : horizons.map(({ label }, i) => ({
    horizon: label,
    'LSTM-GRU': parseFloat((0.45 - i * 0.04).toFixed(3)),
    'Transformer': parseFloat((0.25 + i * 0.05).toFixed(3)),
    'XGBoost': parseFloat((0.30 - i * 0.01).toFixed(3)),
  }));

  return (
    <div className="dashboard animate-fade-in">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
            GNSS ErrorNet Dashboard
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Multi-horizon satellite error prediction and monitoring
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {usingDemoData ? (
              <WifiOff size={16} style={{ color: '#F59E0B' }} />
            ) : (
              <Wifi size={16} style={{ color: '#10B981' }} />
            )}
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: '500' }}>
              {usingDemoData ? 'Demo Mode' : 'API Connected'}
            </span>
          </div>
          {/* Horizon quick selector */}
          <div style={{ display: 'flex', gap: '0.25rem' }}>
            {horizons.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setSelectedHorizon(value)}
                style={{
                  padding: '0.35rem 0.6rem',
                  borderRadius: '0.375rem',
                  border: selectedHorizon === value ? '1.5px solid var(--primary-500)' : '1px solid var(--border-color)',
                  backgroundColor: selectedHorizon === value ? 'var(--primary-100)' : 'transparent',
                  color: selectedHorizon === value ? 'var(--primary-600)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  fontWeight: selectedHorizon === value ? '600' : '400',
                  transition: 'all 0.2s ease'
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid-cols-4" style={{ marginBottom: '2rem' }}>
        {metrics.map((metric, index) => (
          <div 
            key={metric.title} 
            className="card animate-slide-up"
            style={{ animationDelay: `${index * 100}ms`, textAlign: 'center' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div 
                style={{
                  width: '3rem',
                  height: '3rem',
                  borderRadius: '0.75rem',
                  backgroundColor: 'var(--primary-100)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-600)'
                }}
              >
                <metric.icon size={24} />
              </div>
              <div 
                style={{
                  fontSize: '0.8rem',
                  fontWeight: '500',
                  color: metric.changeType === 'positive' ? '#10B981' : '#F59E0B'
                }}
              >
                {metric.change}
              </div>
            </div>
            <h3 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
              {metric.value}
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              {metric.title}
            </p>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {/* Actual vs Predicted */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              Actual vs Predicted ({horizons.find(h => h.value === selectedHorizon)?.label})
            </h2>
            <Activity size={20} style={{ color: 'var(--text-muted)' }} />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="time" stroke="#6B7280" fontSize={11} interval={5} />
              <YAxis stroke="#6B7280" fontSize={11} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Legend />
              <Area type="monotone" dataKey="upper" stroke="none" fill="#8B5CF6" fillOpacity={0.08} name="±2σ" />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#8B5CF6" fillOpacity={0.08} name=" " />
              <Line type="monotone" dataKey="actual" stroke="#F59E0B" strokeWidth={2} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke="#3B82F6" strokeWidth={2} dot={false} name="Predicted" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction Uncertainty */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              GP Uncertainty (±2σ)
            </h2>
            <AlertTriangle size={20} style={{ color: '#F59E0B' }} />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="time" stroke="#6B7280" fontSize={11} interval={5} />
              <YAxis stroke="#6B7280" fontSize={11} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="uncertainty" 
                stroke="#8B5CF6" 
                fill="#8B5CF6" 
                fillOpacity={0.5}
                name="Uncertainty (±2σ)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {/* RMSE per Horizon */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Error by Prediction Horizon
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={horizonCompData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="horizon" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={11} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Legend />
              <Bar dataKey="rmse" fill="#3B82F6" name="RMSE" radius={[4, 4, 0, 0]} />
              <Bar dataKey="mae" fill="#F59E0B" name="MAE" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Ridge Stacker Weights */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Ensemble Weights by Horizon
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={ridgeWeightData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="horizon" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={11} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Legend />
              <Bar dataKey="LSTM-GRU" stackId="a" fill="#3B82F6" radius={[0, 0, 0, 0]} />
              <Bar dataKey="Transformer" stackId="a" fill="#8B5CF6" radius={[0, 0, 0, 0]} />
              <Bar dataKey="XGBoost" stackId="a" fill="#10B981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System Status */}
      <div className="card" style={{ textAlign: 'center' }}>
        <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Pipeline Status
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
          {[
            { label: 'API Server', ok: !usingDemoData },
            { label: 'Models Trained', ok: apiStatus?.has_model || usingDemoData },
            { label: 'Predictions', ok: predictions !== null },
            { label: 'Residual Normality', ok: allNormal },
          ].map(({ label, ok }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {ok ? (
                <CheckCircle2 size={18} style={{ color: '#10B981' }} />
              ) : (
                <Clock size={18} style={{ color: '#F59E0B' }} />
              )}
              <span style={{ color: 'var(--text-primary)', fontSize: '0.9rem', fontWeight: '500' }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;