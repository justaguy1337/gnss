import React, { useState, useEffect, useCallback } from 'react';
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
  Bar,
  ScatterChart,
  Scatter,
  ReferenceLine
} from 'recharts';
import {
  TrendingUp,
  Download,
  Activity,
  Upload
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const horizons = [
  { value: 1, label: '15 min' },
  { value: 2, label: '30 min' },
  { value: 4, label: '1 hour' },
  { value: 8, label: '2 hours' },
  { value: 96, label: '24 hours' },
];

// Approximate inverse error function for Q-Q plot
function inverseErf(x) {
  const a = 0.147;
  const ln = Math.log(1 - x * x);
  const s = Math.sign(x);
  const t = 2 / (Math.PI * a) + ln / 2;
  return s * Math.sqrt(Math.sqrt(t * t - ln / a) - t);
}

const Predictions = () => {
  const [predictions, setPredictions] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [selectedHorizon, setSelectedHorizon] = useState(1);
  const [loading, setLoading] = useState(true);

  const [usingDemoData, setUsingDemoData] = useState(false);

  // Generate flat 0 data if API is not available
  const generateDemoData = useCallback(() => {
    const demoData = {};
    const demoEval = {};

    horizons.forEach(({ value: h }) => {
      const n = 96;
      const preds = Array(n).fill(0);
      const truths = Array(n).fill(0);
      const uncertainties = Array(n).fill(0);
      const residuals = Array(n).fill(0);
      const rmse = 0;
      const mae = 0;

      demoData[h] = {
        predictions: preds,
        ground_truth: truths,
        uncertainties: uncertainties,
        residuals: residuals,
        horizon: h,
        horizon_min: h * 15,
        n_predictions: n,
        base_predictions: {
          lstm_gru: preds,
          transformer: preds,
          xgboost: preds,
        },
        rmse: rmse,
        mae: mae,
      };

      demoEval[h] = {
        horizon: h,
        horizon_min: h * 15,
        rmse: rmse,
        mae: mae,
        r2_score: 0,
        residual_mean: 0,
        residual_std: 0,
        residual_skewness: 0,
        residual_kurtosis: 0,
        shapiro_wilk: {
          p_value: 1.0,
          is_normal: true,
        },
        histogram_data: {
          bin_centers: Array(30).fill(0),
          counts: Array(30).fill(0),
          normal_curve: {
            x: Array(50).fill(0),
            y: Array(50).fill(0),
          },
        },
        qq_data: {
          theoretical: Array(96).fill(0),
          sample: Array(96).fill(0),
        },
      };
    });

    return { predictions: demoData, evaluation: demoEval };
  }, []);

  const [apiStatus, setApiStatus] = useState(null);

  useEffect(() => {
    let first = true;
    const fetchData = async () => {
      try {
        const statusRes = await fetch(`${API_BASE}/status`).catch(() => null);
        if (statusRes?.ok) {
          const status = await statusRes.json();
          setApiStatus(status);
          if (!status.test_data_uploaded) {
            if (first) { setLoading(false); first = false; }
            return;
          }
        }
        const [predRes, evalRes] = await Promise.all([
          fetch(`${API_BASE}/predict/all`),
          fetch(`${API_BASE}/evaluation`),
        ]);

        if (predRes.ok && evalRes.ok) {
          setPredictions(await predRes.json());
          setEvaluation(await evalRes.json());
          setUsingDemoData(false);
        } else {
          throw new Error('API not available');
        }
      } catch {
        if (first) {
          const demo = generateDemoData();
          setPredictions(demo.predictions);
          setEvaluation(demo.evaluation);
          setUsingDemoData(true);
        }
      } finally {
        if (first) { setLoading(false); first = false; }
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [generateDemoData]);

  if (loading) {
    return (
      <div className="animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={48} style={{ color: 'var(--primary-500)', animation: 'pulse 2s infinite' }} />
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>Loading predictions...</p>
        </div>
      </div>
    );
  }

  if (apiStatus?.test_data_uploaded === false) {
    return (
      <div className="animate-fade-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
        <div style={{ textAlign: 'center', maxWidth: '420px' }}>
          <div style={{
            width: '80px', height: '80px', borderRadius: '50%',
            backgroundColor: 'var(--primary-100)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 1.5rem',
          }}>
            <Upload size={36} style={{ color: 'var(--primary-500)' }} />
          </div>
          <h2 style={{ color: 'var(--text-primary)', marginBottom: '0.75rem', fontSize: '1.5rem', fontWeight: '600' }}>
            No Test Data Yet
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: '1.6' }}>
            Upload a test CSV to generate multi-horizon predictions.
          </p>
          <a href="/upload" style={{
            display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
            backgroundColor: 'var(--primary-500)', color: 'white',
            padding: '0.65rem 1.5rem', borderRadius: '0.5rem',
            fontWeight: '500', textDecoration: 'none', fontSize: '0.95rem',
          }}>
            <Upload size={16} /> Go to Data Upload
          </a>
        </div>
      </div>
    );
  }

  const currentPred = predictions?.[String(selectedHorizon)];
  const currentEval = evaluation?.[String(selectedHorizon)];

  // Prepare time-series chart data
  const timeSeriesData = currentPred ? currentPred.predictions.map((pred, i) => ({
    step: i + 1,
    time: `${String(Math.floor(i * 15 / 60)).padStart(2, '0')}:${String((i * 15) % 60).padStart(2, '0')}`,
    predicted: parseFloat(pred.toFixed(4)),
    actual: currentPred.ground_truth ? parseFloat(currentPred.ground_truth[i]?.toFixed(4)) : null,
    upper: parseFloat((pred + (currentPred.uncertainties?.[i] || 0) * 2).toFixed(4)),
    lower: parseFloat((pred - (currentPred.uncertainties?.[i] || 0) * 2).toFixed(4)),
  })) : [];

  // Prepare histogram data
  const histogramData = currentEval?.histogram_data ? currentEval.histogram_data.bin_centers.map((center, i) => ({
    x: parseFloat(center.toFixed(4)),
    count: parseFloat(currentEval.histogram_data.counts[i]?.toFixed(4) || 0),
  })) : [];

  // Prepare Q-Q plot data
  const qqData = currentEval?.qq_data ? currentEval.qq_data.theoretical.map((t, i) => ({
    theoretical: parseFloat(t.toFixed(3)),
    sample: parseFloat(currentEval.qq_data.sample[i]?.toFixed(3) || 0),
  })) : [];

  // Horizon comparison data
  const horizonCompData = horizons.map(({ value: h, label }) => {
    const ev = evaluation?.[String(h)];
    return {
      horizon: label,
      rmse: ev ? parseFloat(ev.rmse.toFixed(5)) : 0,
      mae: ev ? parseFloat(ev.mae.toFixed(5)) : 0,
      r2: ev ? parseFloat(ev.r2_score.toFixed(4)) : 0,
    };
  });

  const downloadCSV = () => {
    if (!currentPred) return;
    let csv = 'step,time,predicted,actual,uncertainty\n';
    currentPred.predictions.forEach((pred, i) => {
      const time = `${String(Math.floor(i * 15 / 60)).padStart(2, '0')}:${String((i * 15) % 60).padStart(2, '0')}`;
      const actual = currentPred.ground_truth?.[i] ?? '';
      const unc = currentPred.uncertainties?.[i] ?? '';
      csv += `${i + 1},${time},${pred.toFixed(6)},${actual ? actual.toFixed(6) : ''},${unc ? unc.toFixed(6) : ''}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_h${selectedHorizon}_${selectedHorizon * 15}min.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
            Day 8 Predictions
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Multi-horizon error predictions with uncertainty quantification
            {usingDemoData && (
              <span style={{ color: 'var(--accent-500)', marginLeft: '0.5rem', fontSize: '0.8rem' }}>
                (Demo data — start API server for real predictions)
              </span>
            )}
          </p>
        </div>
        <button
          onClick={downloadCSV}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            backgroundColor: 'var(--primary-600)', color: 'white',
            padding: '0.75rem 1.25rem', borderRadius: '0.5rem',
            border: 'none', cursor: 'pointer', fontWeight: '500'
          }}
        >
          <Download size={18} />
          Export CSV
        </button>
      </div>

      {/* Horizon Selector */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h3 className="font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Prediction Horizon</h3>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          {horizons.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setSelectedHorizon(value)}
              style={{
                padding: '0.5rem 1.25rem',
                borderRadius: '0.5rem',
                border: selectedHorizon === value ? '2px solid var(--primary-500)' : '1px solid var(--border-color)',
                backgroundColor: selectedHorizon === value ? 'var(--primary-100)' : 'transparent',
                color: selectedHorizon === value ? 'var(--primary-600)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontWeight: selectedHorizon === value ? '600' : '400',
                transition: 'all 0.2s ease'
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid-cols-4" style={{ marginBottom: '1.5rem' }}>
        {[
          { label: 'RMSE', value: currentEval?.rmse?.toFixed(5) || '—', color: 'var(--primary-600)' },
          { label: 'MAE', value: currentEval?.mae?.toFixed(5) || '—', color: 'var(--accent-600)' },
          { label: 'R² Score', value: currentEval?.r2_score?.toFixed(4) || '—', color: '#10B981' },
          {
            label: 'Normality (S-W)',
            value: currentEval?.shapiro_wilk?.is_normal ? '✓ Normal' : '✗ Non-normal',
            color: currentEval?.shapiro_wilk?.is_normal ? '#10B981' : '#EF4444',
            sub: `p = ${currentEval?.shapiro_wilk?.p_value?.toFixed(4) || '—'}`
          },
        ].map((metric, i) => (
          <div key={i} className="card animate-slide-up" style={{ animationDelay: `${i * 80}ms`, textAlign: 'center' }}>
            <div className="text-2xl font-bold mb-1" style={{ color: metric.color }}>{metric.value}</div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>{metric.label}</div>
            {metric.sub && <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.25rem' }}>{metric.sub}</div>}
          </div>
        ))}
      </div>

      {/* Time-Series Chart */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
            Actual vs Predicted — {horizons.find(h => h.value === selectedHorizon)?.label} Horizon
          </h2>
          <TrendingUp size={20} style={{ color: 'var(--text-muted)' }} />
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis dataKey="time" stroke="#6B7280" fontSize={11} interval={11} />
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
            <Area type="monotone" dataKey="upper" stroke="none" fill="#8B5CF6" fillOpacity={0.1} name="±2σ Upper" />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#8B5CF6" fillOpacity={0.1} name="±2σ Lower" />
            <Line type="monotone" dataKey="actual" stroke="#F59E0B" strokeWidth={2} dot={false} name="Actual" />
            <Line type="monotone" dataKey="predicted" stroke="#3B82F6" strokeWidth={2} dot={false} name="Predicted" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Distribution & Q-Q Plot */}
      <div className="grid-cols-2" style={{ marginBottom: '1.5rem' }}>
        {/* Residual Histogram */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
            Residual Distribution
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="x" stroke="#6B7280" fontSize={10} />
              <YAxis stroke="#6B7280" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Bar dataKey="count" fill="#8B5CF6" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Q-Q Plot */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
            Q-Q Plot (Normal)
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis type="number" dataKey="theoretical" name="Theoretical" stroke="#6B7280" fontSize={10} />
              <YAxis type="number" dataKey="sample" name="Sample" stroke="#6B7280" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <ReferenceLine
                segment={[{ x: -3, y: -3 }, { x: 3, y: 3 }]}
                stroke="#EF4444"
                strokeDasharray="5 5"
                strokeWidth={2}
              />
              <Scatter data={qqData} fill="#3B82F6" fillOpacity={0.6} r={3} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Horizon Comparison */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Performance Across Horizons
        </h2>
        <ResponsiveContainer width="100%" height={300}>
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
    </div>
  );
};

export default Predictions;
