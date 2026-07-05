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
  Activity
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

  // Generate demo data if API is not available
  const generateDemoData = useCallback(() => {
    const demoData = {};
    const demoEval = {};

    horizons.forEach(({ value: h }) => {
      const n = 96;
      const preds = [];
      const truths = [];
      const uncertainties = [];
      const residuals = [];

      for (let i = 0; i < n; i++) {
        const t = (i / n) * 2 * Math.PI;
        const truth = 2 * Math.sin(t) + 0.5 * Math.cos(2 * t) + (Math.random() - 0.5) * 0.3;
        const noise = (Math.random() - 0.5) * (0.1 + h * 0.01);
        const pred = truth + noise;
        const unc = 0.1 + h * 0.02 + Math.random() * 0.05;
        preds.push(pred);
        truths.push(truth);
        uncertainties.push(unc);
        residuals.push(truth - pred);
      }

      demoData[h] = {
        predictions: preds,
        ground_truth: truths,
        uncertainties: uncertainties,
        residuals: residuals,
        horizon: h,
        horizon_min: h * 15,
        n_predictions: n,
        base_predictions: {
          lstm_gru: preds.map(p => p + (Math.random() - 0.5) * 0.2),
          transformer: preds.map(p => p + (Math.random() - 0.5) * 0.15),
          xgboost: preds.map(p => p + (Math.random() - 0.5) * 0.25),
        },
        rmse: 0.05 + h * 0.01,
        mae: 0.04 + h * 0.008,
      };

      const resSorted = [...residuals].sort((a, b) => a - b);
      const mean = residuals.reduce((a, b) => a + b, 0) / n;
      const std = Math.sqrt(residuals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);

      demoEval[h] = {
        horizon: h,
        horizon_min: h * 15,
        rmse: demoData[h].rmse,
        mae: demoData[h].mae,
        r2_score: 0.95 - h * 0.005,
        residual_mean: mean,
        residual_std: std,
        residual_skewness: 0.05 + (Math.random() - 0.5) * 0.1,
        residual_kurtosis: -0.1 + (Math.random() - 0.5) * 0.2,
        shapiro_wilk: {
          p_value: 0.15 + Math.random() * 0.5,
          is_normal: true,
        },
        histogram_data: {
          bin_centers: Array.from({ length: 30 }, (_, i) => (i - 15) * std / 5 + mean),
          counts: Array.from({ length: 30 }, (_, i) => {
            const x = (i - 15) / 5;
            return Math.exp(-x * x / 2) / (std * Math.sqrt(2 * Math.PI)) + Math.random() * 0.05;
          }),
          normal_curve: {
            x: Array.from({ length: 50 }, (_, i) => (i - 25) * std / 8 + mean),
            y: Array.from({ length: 50 }, (_, i) => {
              const x = (i - 25) / 8;
              return Math.exp(-x * x / 2) / (std * Math.sqrt(2 * Math.PI));
            }),
          },
        },
        qq_data: {
          theoretical: resSorted.map((_, i) => {
            const p = (i + 0.5) / n;
            return Math.sqrt(2) * inverseErf(2 * p - 1);
          }),
          sample: resSorted.map(r => (r - mean) / (std || 1)),
        },
      };
    });

    return { predictions: demoData, evaluation: demoEval };
  }, []);

  useEffect(() => {
    let first = true;
    const fetchData = async () => {
      try {
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

  const currentPred = predictions?.[selectedHorizon];
  const currentEval = evaluation?.[selectedHorizon];

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
    const ev = evaluation?.[h];
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
