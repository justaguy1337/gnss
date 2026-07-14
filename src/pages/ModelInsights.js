import React, { useState, useEffect, useCallback } from 'react';
import { Brain, Cpu, Zap, Target, X, GitMerge } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';

const API_BASE = 'http://localhost:8000/api';

const ModelInsights = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [usingDemoData, setUsingDemoData] = useState(false);

  const generateDemoEval = useCallback(() => {
    const horizons = [1, 2, 4, 8, 16];
    const ev = {};
    horizons.forEach(h => {
      ev[h] = {
        horizon: h, horizon_min: h * 15,
        rmse: 0,
        mae: 0,
        r2_score: 0,
        shapiro_wilk: { p_value: 1.0, is_normal: true },
        residual_mean: 0,
        residual_std: 0,
      };
    });
    return ev;
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [evalRes, modelsRes] = await Promise.all([
          fetch(`${API_BASE}/evaluation`).catch(() => null),
          fetch(`${API_BASE}/models`).catch(() => null),
        ]);
        if (evalRes?.ok) {
          setEvaluation(await evalRes.json());
          if (modelsRes?.ok) setModelsInfo(await modelsRes.json());
          setUsingDemoData(false);
        } else throw new Error();
      } catch {
        setEvaluation(generateDemoEval());
        setUsingDemoData(true);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [generateDemoEval]);

  // Actual models from the paper
  const models = [
    {
      name: 'LSTM-GRU',
      icon: Brain,
      role: 'Base Model 1',
      description: 'Hybrid recurrent network for short-term drift capture (Paper §III-C)',
      architecture: 'Input(96×1) → LSTM(64, 2 layers) → GRU(64, 1 layer) → Dense(32) → ReLU → Dense(1)',
      details: [
        'LSTM captures long-term dependencies via forget/input/output gates',
        'GRU refines with lighter reset/update gating',
        'Adam optimizer + gradient clipping (max norm = 1.0)',
        'LR scheduling: ReduceOnPlateau with patience=5',
        'Early stopping with patience=10',
        'Target normalization via StandardScaler',
      ],
      color: '#3B82F6',
    },
    {
      name: 'Transformer',
      icon: Zap,
      role: 'Base Model 2',
      description: 'Self-attention encoder for long-range orbital cycle patterns (Paper §III-D)',
      architecture: 'Input(96×1) → Linear(1→64) + LearnedPosEncoding → TransformerEncoder(2L, 4H) → LastToken → Dense(1)',
      details: [
        'Learned positional embedding (nn.Embedding, not sinusoidal)',
        'Multi-head self-attention scans all timestamp pairs',
        'Captures orbital period harmonics across 96 timesteps',
        'Same training protocol as LSTM-GRU',
        'd_model=64, n_heads=4, FFN dim=128',
      ],
      color: '#8B5CF6',
    },
    {
      name: 'XGBoost',
      icon: Target,
      role: 'Base Model 3',
      description: 'Gradient boosted trees on engineered tabular features (Paper §III-E)',
      architecture: 'X_tab (lags + rolling stats + cyclic + diffs + horizon h) → XGBRegressor → prediction',
      details: [
        'Single unified model — horizon h is an input feature',
        'Lag features at t-1, t-2, t-4, t-8, t-96',
        'Rolling mean/std/max/min over 3-hour window',
        'Cyclic sin/cos for daily and half-daily patterns',
        '1st & 2nd order differences (rate + acceleration)',
        'Regularized objective with tree complexity penalty',
        'max_depth=6, n_estimators=500, lr=0.05',
      ],
      color: '#10B981',
    },
    {
      name: 'Ridge Stacker',
      icon: GitMerge,
      role: 'Meta-Learner',
      description: 'Per-horizon weighted blending of base model predictions (Paper §III-F)',
      architecture: '[p_LSTM, p_Transformer, p_XGB] → Ridge(α=1.0) → ŷ_h per horizon',
      details: [
        'Separate Ridge regression per forecast horizon h',
        'Learns adaptive weights w₁, w₂, w₃ per horizon',
        'Short horizons: LSTM-GRU typically gets higher weight',
        'Long horizons: Transformer typically dominates',
        'Trained on out-of-fold validation predictions',
        'L2 regularization prevents weight explosion',
      ],
      color: '#F59E0B',
    },
    {
      name: 'Gaussian Process',
      icon: Cpu,
      role: 'Residual Corrector',
      description: 'GP with Matérn + Periodic kernel for residual modeling & uncertainty (Paper §III-G)',
      architecture: 'r = y - ŷ_stacker → GP(k_Matérn(ν=2.5) + k_Periodic) → μ_GP, σ_GP',
      details: [
        'Models stacker residuals r_h = y_h - ŷ_h',
        'Composite kernel: Matérn(ν=2.5) + Periodic',
        'Matérn: rough but continuous residual patterns',
        'Periodic: repeating orbital cycle signals',
        'Kernel hyperparameters via log marginal likelihood',
        'Outputs: mean correction μ_GP + uncertainty σ_GP',
        'Final: ŷ_final = ŷ_h + μ_GP',
        'Key: ensures approximately Gaussian residuals',
      ],
      color: '#EC4899',
    },
  ];

  // Compute average metrics from evaluation
  const avgR2 = evaluation
    ? Object.values(evaluation).reduce((s, ev) => s + (ev.r2_score || 0), 0) / Object.keys(evaluation).length
    : 0;
  const avgRMSE = evaluation
    ? Object.values(evaluation).reduce((s, ev) => s + (ev.rmse || 0), 0) / Object.keys(evaluation).length
    : 0;

  // Ridge weights per horizon
  const weightsData = modelsInfo
    ? [1, 2, 4, 8, 16].map(h => {
        const w = modelsInfo.horizons?.[String(h)]?.weights || {};
        return {
          horizon: `${h * 15}min`,
          'LSTM-GRU': parseFloat((w['LSTM-GRU'] ?? 0).toFixed(3)),
          'Transformer': parseFloat((w['Transformer'] ?? 0).toFixed(3)),
          'XGBoost': parseFloat((w['XGBoost'] ?? 0).toFixed(3)),
        };
      })
    : [1, 2, 4, 8, 16].map(h => ({
        horizon: `${h * 15}min`,
        'LSTM-GRU': 0,
        'Transformer': 0,
        'XGBoost': 0,
      }));



  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Model Insights
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Stacked ensemble architecture: 3 base models → Ridge stacker → GP residual correction
          {usingDemoData && (
            <span style={{ color: 'var(--accent-500)', marginLeft: '0.5rem', fontSize: '0.8rem' }}>
              (Demo data)
            </span>
          )}
        </p>
      </div>

      {/* Pipeline Flow Visualization */}
      <div className="card" style={{ marginBottom: '2rem', padding: '1.5rem', textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {models.map((model, i) => (
            <React.Fragment key={model.name}>
              <div
                onClick={() => setSelectedModel(model)}
                style={{
                  padding: '0.5rem 1rem',
                  borderRadius: '0.5rem',
                  backgroundColor: model.color + '20',
                  border: `1.5px solid ${model.color}`,
                  color: model.color,
                  fontWeight: '600',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  transition: 'transform 0.2s ease',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.05)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
              >
                {model.name}
              </div>
              {i < models.length - 1 && (
                <span style={{ color: 'var(--text-muted)', fontSize: '1.2rem' }}>→</span>
              )}
            </React.Fragment>
          ))}
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.75rem' }}>
          Click any component to view architecture details
        </p>
      </div>

      {/* Model Cards */}
      <div className="grid-cols-3" style={{ marginBottom: '2rem' }}>
        {models.slice(0, 3).map((model, index) => (
          <div 
            key={model.name} 
            className="card animate-slide-up" 
            style={{ 
              animationDelay: `${index * 100}ms`,
              cursor: 'pointer',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              borderTop: `3px solid ${model.color}`
            }}
            onClick={() => setSelectedModel(model)}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = `0 6px 24px ${model.color}30`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <div style={{
                width: '2.5rem', height: '2.5rem', borderRadius: '0.5rem',
                backgroundColor: model.color + '20',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: model.color
              }}>
                <model.icon size={20} />
              </div>
              <div>
                <h3 className="font-semibold" style={{ color: 'var(--text-primary)', fontSize: '1rem' }}>
                  {model.name}
                </h3>
                <span style={{ color: model.color, fontSize: '0.7rem', fontWeight: '600' }}>
                  {model.role}
                </span>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: '1.4' }}>
              {model.description}
            </p>
          </div>
        ))}
      </div>

      {/* Meta-learner + GP cards */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {models.slice(3).map((model, index) => (
          <div 
            key={model.name} 
            className="card animate-slide-up" 
            style={{ 
              animationDelay: `${(index + 3) * 100}ms`,
              cursor: 'pointer',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease',
              borderTop: `3px solid ${model.color}`
            }}
            onClick={() => setSelectedModel(model)}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = `0 6px 24px ${model.color}30`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <div style={{
                width: '2.5rem', height: '2.5rem', borderRadius: '0.5rem',
                backgroundColor: model.color + '20',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: model.color
              }}>
                <model.icon size={20} />
              </div>
              <div>
                <h3 className="font-semibold" style={{ color: 'var(--text-primary)', fontSize: '1rem' }}>
                  {model.name}
                </h3>
                <span style={{ color: model.color, fontSize: '0.7rem', fontWeight: '600' }}>
                  {model.role}
                </span>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', lineHeight: '1.4' }}>
              {model.description}
            </p>
          </div>
        ))}
      </div>

      {/* Ensemble Performance + Ridge Weights */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        <div className="card" style={{ textAlign: 'center' }}>
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Ensemble Performance
          </h2>
          <div className="text-4xl font-bold mb-2" style={{ color: 'var(--primary-600)' }}>
            {(avgR2 * 100).toFixed(1)}%
          </div>
          <div style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Average R² Score</div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem' }}>
            <div>
              <div className="text-xl font-bold" style={{ color: '#10B981' }}>
                {avgRMSE.toFixed(4)}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Avg RMSE</div>
            </div>
            <div>
              <div className="text-xl font-bold" style={{ color: '#8B5CF6' }}>
                {evaluation ? Object.values(evaluation).filter(ev => ev.shapiro_wilk?.is_normal).length : 0}/{evaluation ? Object.keys(evaluation).length : 0}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Normal Residuals</div>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Ridge Stacker Weights by Horizon
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={weightsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="horizon" stroke="#6B7280" fontSize={11} />
              <YAxis stroke="#6B7280" fontSize={11} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                  borderRadius: '8px', color: 'var(--text-primary)'
                }}
              />
              <Legend />
              <Bar dataKey="LSTM-GRU" stackId="a" fill="#3B82F6" />
              <Bar dataKey="Transformer" stackId="a" fill="#8B5CF6" />
              <Bar dataKey="XGBoost" stackId="a" fill="#10B981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Modal for model details */}
      {selectedModel && (
        <div 
          style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 1000, padding: '1rem'
          }}
          onClick={() => setSelectedModel(null)}
        >
          <div 
            className="card"
            style={{
              width: '90%', maxWidth: '700px', maxHeight: '80vh',
              overflow: 'auto', position: 'relative',
              borderTop: `4px solid ${selectedModel.color}`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{
                  width: '2.5rem', height: '2.5rem', borderRadius: '0.5rem',
                  backgroundColor: selectedModel.color + '20',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: selectedModel.color
                }}>
                  <selectedModel.icon size={20} />
                </div>
                <div>
                  <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
                    {selectedModel.name}
                  </h2>
                  <span style={{ color: selectedModel.color, fontSize: '0.8rem', fontWeight: '600' }}>
                    {selectedModel.role}
                  </span>
                </div>
              </div>
              <button
                onClick={() => setSelectedModel(null)}
                style={{
                  background: 'none', border: 'none', color: 'var(--text-secondary)',
                  cursor: 'pointer', padding: '0.5rem'
                }}
              >
                <X size={24} />
              </button>
            </div>

            {/* Architecture */}
            <div style={{
              backgroundColor: 'var(--surface-alt)', borderRadius: '0.5rem',
              padding: '1rem', marginBottom: '1.5rem',
              fontFamily: 'monospace', fontSize: '0.8rem',
              color: 'var(--text-primary)', lineHeight: '1.6',
              borderLeft: `3px solid ${selectedModel.color}`
            }}>
              {selectedModel.architecture}
            </div>

            {/* Details */}
            <h3 className="font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
              Implementation Details
            </h3>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {selectedModel.details.map((detail, i) => (
                <li key={i} style={{
                  display: 'flex', alignItems: 'flex-start', gap: '0.5rem',
                  padding: '0.4rem 0', color: 'var(--text-secondary)', fontSize: '0.85rem'
                }}>
                  <span style={{ color: selectedModel.color, fontWeight: 'bold', flexShrink: 0 }}>•</span>
                  {detail}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelInsights;