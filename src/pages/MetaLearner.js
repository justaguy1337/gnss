import React, { useState, useEffect, useCallback } from 'react';
import './MetaLearner.css';

const API_BASE = 'http://localhost:8000/api';

const MetaLearner = () => {
  const [showResults, setShowResults] = useState(false);
  const [isComputing, setIsComputing] = useState(false);
  const [showAnimations, setShowAnimations] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [usingDemoData, setUsingDemoData] = useState(false);

  // Actual pipeline components from the paper
  const models = [
    { name: 'LSTM-GRU', role: 'Base Model', color: '#3B82F6', position: { top: '20%', left: '25%' } },
    { name: 'Transformer', role: 'Base Model', color: '#8B5CF6', position: { top: '20%', left: '75%' } },
    { name: 'XGBoost', role: 'Base Model', color: '#10B981', position: { top: '55%', left: '85%' } },
    { name: 'GP Corrector', role: 'Residual', color: '#EC4899', position: { top: '80%', left: '65%' } },
    { name: 'Ridge Stacker', role: 'Meta-Learner', color: '#F59E0B', position: { top: '80%', left: '35%' } },
  ];

  // SVG coordinates matching the CSS positions (in 1000x600 viewBox)
  const svgPositions = [
    { x: 250, y: 120 },   // LSTM-GRU
    { x: 750, y: 120 },   // Transformer
    { x: 850, y: 330 },   // XGBoost
    { x: 650, y: 480 },   // GP Corrector
    { x: 350, y: 480 },   // Ridge Stacker
  ];

  const generateDemoData = useCallback(() => {
    const horizons = [1, 2, 4, 8, 96];
    const ev = {};
    horizons.forEach(h => {
      ev[h] = {
        horizon: h, horizon_min: h * 15,
        rmse: 0.04 + h * 0.008,
        mae: 0.03 + h * 0.006,
        r2_score: 0.97 - h * 0.003,
        shapiro_wilk: { p_value: 0.25, is_normal: true },
      };
    });

    const mi = {
      base_models: ['LSTM-GRU', 'Transformer', 'XGBoost'],
      meta_learner: 'Ridge Regression (per-horizon)',
      residual_model: 'Gaussian Process (Matérn 2.5 + Periodic)',
      horizons: {},
    };
    horizons.forEach((h, i) => {
      mi.horizons[h] = {
        weights: {
          'LSTM-GRU': 0.45 - i * 0.04,
          'Transformer': 0.25 + i * 0.05,
          'XGBoost': 0.30 - i * 0.01,
        },
        alpha: 1.0,
      };
    });

    return { evaluation: ev, modelsInfo: mi };
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [evalRes, modelsRes] = await Promise.all([
          fetch(`${API_BASE}/evaluation`).catch(() => null),
          fetch(`${API_BASE}/models`).catch(() => null),
        ]);
        if (evalRes?.ok && modelsRes?.ok) {
          setEvaluation(await evalRes.json());
          setModelsInfo(await modelsRes.json());
          setUsingDemoData(false);
        } else throw new Error();
      } catch {
        const demo = generateDemoData();
        setEvaluation(demo.evaluation);
        setModelsInfo(demo.modelsInfo);
        setUsingDemoData(true);
      }
    };
    fetchData();
  }, [generateDemoData]);

  // Compute real metrics
  const avgR2 = evaluation
    ? (Object.values(evaluation).reduce((s, ev) => s + (ev.r2_score || 0), 0) / Object.keys(evaluation).length * 100)
    : 0;
  const avgRMSE = evaluation
    ? Object.values(evaluation).reduce((s, ev) => s + (ev.rmse || 0), 0) / Object.keys(evaluation).length
    : 0;
  const bestR2 = evaluation
    ? Math.max(...Object.values(evaluation).map(ev => ev.r2_score || 0)) * 100
    : 0;

  // Per-model "contribution" from Ridge weights (avg across horizons)
  const getModelScore = (modelName) => {
    if (!modelsInfo?.horizons) return 33;
    const horizonKeys = Object.keys(modelsInfo.horizons);
    if (horizonKeys.length === 0) return 33;
    const sum = horizonKeys.reduce((s, h) => {
      const w = modelsInfo.horizons[h]?.weights?.[modelName] || 0.33;
      return s + w;
    }, 0);
    return (sum / horizonKeys.length * 100).toFixed(1);
  };

  const modelScores = {
    'LSTM-GRU': getModelScore('LSTM-GRU'),
    'Transformer': getModelScore('Transformer'),
    'XGBoost': getModelScore('XGBoost'),
    'Ridge Stacker': avgR2.toFixed(1),
    'GP Corrector': bestR2.toFixed(1),
  };

  const handleCompute = () => {
    setIsComputing(true);
    setShowAnimations(true);

    setTimeout(() => {
      setIsComputing(false);
      setShowResults(true);
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({
          behavior: 'smooth'
        });
      }, 300);
    }, 6000);
  };

  const resetVisualization = () => {
    setShowResults(false);
    setIsComputing(false);
    setShowAnimations(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Results section data
  const resultsModels = [
    { name: 'LSTM-GRU', score: modelScores['LSTM-GRU'], color: '#3B82F6', label: 'Avg Weight' },
    { name: 'Transformer', score: modelScores['Transformer'], color: '#8B5CF6', label: 'Avg Weight' },
    { name: 'XGBoost', score: modelScores['XGBoost'], color: '#10B981', label: 'Avg Weight' },
    { name: 'Ridge Ensemble', score: avgR2.toFixed(1), color: '#F59E0B', label: 'R² %' },
    { name: 'GP Final', score: bestR2.toFixed(1), color: '#EC4899', label: 'Best R² %' },
  ];

  // Per-horizon weight table
  const horizonWeights = modelsInfo?.horizons
    ? Object.entries(modelsInfo.horizons).map(([h, data]) => ({
        horizon: `${parseInt(h) * 15}min`,
        lstm: (data.weights?.['LSTM-GRU'] || 0.33).toFixed(3),
        transformer: (data.weights?.['Transformer'] || 0.33).toFixed(3),
        xgboost: (data.weights?.['XGBoost'] || 0.33).toFixed(3),
      }))
    : [];

  return (
    <div className="meta-learner-simple">
      {/* Header */}
      <div className="simple-header">
        <h1>Stacked Ensemble Meta Learner</h1>
        <p>
          Ridge stacker blends 3 base models → GP residual correction for GNSS error prediction
          {usingDemoData && (
            <span style={{ color: '#F59E0B', marginLeft: '0.5rem', fontSize: '0.8rem' }}>
              (Demo data — start API for real metrics)
            </span>
          )}
        </p>
      </div>

      {/* Simple Mindmap */}
      <div className="simple-mindmap">
        {/* SVG Connections */}
        <svg className="connections-svg" viewBox="0 0 1000 600">
          {svgPositions.map((pos, i) => (
            <React.Fragment key={i}>
              <line
                x1={pos.x} y1={pos.y} x2={500} y2={300}
                stroke={models[i].color} strokeWidth="2"
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`}
                style={{ animationDelay: `${i * 0.3}s` }}
              />
            </React.Fragment>
          ))}

          {/* Animated particles */}
          {showAnimations && svgPositions.map((pos, i) => (
            <circle key={`p${i}`} r="4" fill={models[i].color} className="connection-particle">
              <animateMotion dur="2s" repeatCount="indefinite" begin={`${i * 0.3}s`}>
                <mpath href={`#mpath${i}`} />
              </animateMotion>
            </circle>
          ))}

          <defs>
            {svgPositions.map((pos, i) => (
              <path key={`path${i}`} id={`mpath${i}`} d={`M ${pos.x} ${pos.y} L 500 300`} />
            ))}
          </defs>
        </svg>

        {/* Central Meta Learner */}
        <div className="meta-center">
          <div className={`center-circle ${isComputing ? 'computing' : ''}`}>
            <h3>Ensemble</h3>
            <p>Stacked</p>
            {showResults && (
              <div className="final-result">
                <strong>{avgR2.toFixed(1)}%</strong>
              </div>
            )}
            {isComputing && (
              <div className="computing-indicator">
                <div className="loading-spinner"></div>
                <span>Blending...</span>
              </div>
            )}
          </div>
        </div>

        {/* Model Nodes */}
        {models.map((model) => (
          <div
            key={model.name}
            className="model-circle"
            style={{
              ...model.position,
              borderColor: model.color
            }}
          >
            <h4>{model.name}</h4>
            <span style={{ color: model.color }}>
              {modelScores[model.name]}%
            </span>
            <span style={{ fontSize: '0.65rem', color: '#999', marginTop: '0.1rem' }}>
              {model.role}
            </span>
          </div>
        ))}

        {/* Compute Button */}
        {!isComputing && !showResults && (
          <button className="simple-compute-btn" onClick={handleCompute}>
            Compute Ensemble Accuracy
          </button>
        )}

        {/* Reset Button */}
        {(showResults || isComputing) && (
          <button className="simple-reset-btn" onClick={resetVisualization}>
            Reset
          </button>
        )}
      </div>

      {/* Results Section */}
      {showResults && (
        <div id="results-section" className="simple-results">
          <h2>Ensemble Results</h2>

          <div className="results-cards">
            {/* Model Weights / Scores */}
            <div className="result-card">
              <h3>Component Performance</h3>
              {resultsModels.map((model) => (
                <div key={model.name} className="model-result">
                  <span className="model-name">{model.name}</span>
                  <div className="accuracy-bar">
                    <div
                      className="accuracy-fill"
                      style={{
                        width: `${model.score}%`,
                        backgroundColor: model.color
                      }}
                    ></div>
                  </div>
                  <span className="accuracy-text">{model.score}%</span>
                </div>
              ))}
            </div>

            {/* Ensemble Metrics */}
            <div className="result-card">
              <h3>Final Metrics</h3>
              <div className="final-metrics">
                <div className="metric">
                  <strong>{avgR2.toFixed(1)}%</strong>
                  <span>Avg R² Score</span>
                </div>
                <div className="metric">
                  <strong>{avgRMSE.toFixed(4)}</strong>
                  <span>Avg RMSE</span>
                </div>
                <div className="metric">
                  <strong>{evaluation ? Object.keys(evaluation).length : 0}</strong>
                  <span>Horizons</span>
                </div>
              </div>
            </div>

            {/* Ridge Weights Table */}
            <div className="result-card full-width">
              <h3>Ridge Stacker Weights per Horizon</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Horizon</th>
                      <th style={{ ...thStyle, color: '#3B82F6' }}>LSTM-GRU</th>
                      <th style={{ ...thStyle, color: '#8B5CF6' }}>Transformer</th>
                      <th style={{ ...thStyle, color: '#10B981' }}>XGBoost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {horizonWeights.map((row, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{row.horizon}</td>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{
                              width: `${parseFloat(row.lstm) * 200}px`, height: '8px',
                              backgroundColor: '#3B82F6', borderRadius: '4px', minWidth: '4px'
                            }}></div>
                            {row.lstm}
                          </div>
                        </td>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{
                              width: `${parseFloat(row.transformer) * 200}px`, height: '8px',
                              backgroundColor: '#8B5CF6', borderRadius: '4px', minWidth: '4px'
                            }}></div>
                            {row.transformer}
                          </div>
                        </td>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div style={{
                              width: `${parseFloat(row.xgboost) * 200}px`, height: '8px',
                              backgroundColor: '#10B981', borderRadius: '4px', minWidth: '4px'
                            }}></div>
                            {row.xgboost}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Per-horizon RMSE chart (SVG) */}
            <div className="result-card full-width">
              <h3>RMSE by Prediction Horizon</h3>
              <div className="line-chart-container">
                <svg className="line-chart" viewBox="0 0 800 300">
                  {/* Grid */}
                  <defs>
                    <pattern id="grid" width="160" height="60" patternUnits="userSpaceOnUse">
                      <path d="M 160 0 L 0 0 0 60" fill="none" stroke="#e0e0e0" strokeWidth="1"/>
                    </pattern>
                  </defs>
                  <rect width="800" height="260" fill="url(#grid)" />

                  {/* Plot data */}
                  {evaluation && (() => {
                    const entries = Object.values(evaluation).sort((a, b) => a.horizon - b.horizon);
                    const maxRMSE = Math.max(...entries.map(e => e.rmse)) * 1.2;
                    const points = entries.map((e, i) => ({
                      x: 80 + i * (640 / (entries.length - 1 || 1)),
                      y: 240 - (e.rmse / maxRMSE) * 220,
                      rmse: e.rmse.toFixed(5),
                      horizon: `${e.horizon * 15}min`,
                    }));

                    const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

                    return (
                      <>
                        <path d={pathD} fill="none" stroke="#3B82F6" strokeWidth="3" className="line-path" />
                        {points.map((p, i) => (
                          <React.Fragment key={i}>
                            <circle cx={p.x} cy={p.y} r="6" fill={models[Math.min(i, models.length - 1)].color} className="data-point" />
                            <text x={p.x} y={p.y - 15} textAnchor="middle" className="accuracy-value-text">{p.rmse}</text>
                            <text x={p.x} y={275} textAnchor="middle" className="chart-label">{p.horizon}</text>
                          </React.Fragment>
                        ))}
                      </>
                    );
                  })()}
                </svg>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const thStyle = {
  textAlign: 'left', padding: '0.75rem 1rem',
  borderBottom: '2px solid #e0e0e0', fontWeight: '600', color: '#333'
};
const tdStyle = {
  padding: '0.6rem 1rem', borderBottom: '1px solid #f0f0f0', color: '#555'
};

export default MetaLearner;