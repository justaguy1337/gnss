import React, { useState } from 'react';
import { Brain, Cpu, Network, Zap, Target, Layers, X, TrendingUp } from 'lucide-react';
import { ResponsiveLine } from '@nivo/line';

const ModelInsights = () => {
  const [selectedModel, setSelectedModel] = useState(null);

  const models = [
    { name: 'LSTM-GRU', icon: Brain, accuracy: '91.2%', description: 'Hybrid recurrent neural network' },
    { name: 'TimeGPT', icon: Zap, accuracy: '94.7%', description: 'Transformer-based foundation model' },
    { name: 'XGBoost', icon: Target, accuracy: '89.5%', description: 'Gradient boosting framework' },
    { name: 'Gaussian Process', icon: Cpu, accuracy: '87.3%', description: 'Probabilistic model' },
    { name: 'GAN', icon: Network, accuracy: '92.8%', description: 'Generative Adversarial Network' },
    { name: 'TCN', icon: Layers, accuracy: '90.2%', description: 'Temporal Convolutional Network' },
  ];

  // Generate dummy prediction data for each model
  const generatePredictionData = (modelName, accuracy) => {
    const baseAccuracy = parseFloat(accuracy.replace('%', ''));
    const points = [];
    
    for (let i = 0; i < 20; i++) {
      const variance = (Math.random() - 0.5) * 10; // ±5% variance
      const prediction = Math.max(0, Math.min(100, baseAccuracy + variance));
      const actual = Math.max(0, Math.min(100, baseAccuracy + (Math.random() - 0.5) * 8));
      
      points.push({
        day: i + 1,
        prediction: prediction.toFixed(1),
        actual: actual.toFixed(1)
      });
    }
    
    return points;
  };

  const handleModelClick = (model) => {
    setSelectedModel({
      ...model,
      predictions: generatePredictionData(model.name, model.accuracy)
    });
  };

  const closeModal = () => {
    setSelectedModel(null);
  };

  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Model Insights
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Understanding the hybrid ML framework for satellite error prediction
        </p>
      </div>

      <div className="grid-cols-3">
        {models.map((model, index) => (
          <div 
            key={model.name} 
            className="card animate-slide-up" 
            style={{ 
              animationDelay: `${index * 100}ms`,
              cursor: 'pointer',
              transition: 'transform 0.2s ease, box-shadow 0.2s ease'
            }}
            onClick={() => handleModelClick(model)}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div 
                style={{
                  width: '2.5rem',
                  height: '2.5rem',
                  backgroundColor: 'var(--primary-100)',
                  borderRadius: '0.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-600)'
                }}
              >
                <model.icon size={20} />
              </div>
              <div>
                <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {model.name}
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  {model.description}
                </p>
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div className="text-2xl font-bold" style={{ color: 'var(--primary-600)' }}>
                {model.accuracy}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                Accuracy
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: '2rem', textAlign: 'center' }}>
        <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Ensemble Performance
        </h2>
        <div className="text-5xl font-bold mb-2" style={{ color: 'var(--accent-500)' }}>
          96.1%
        </div>
        <div style={{ color: 'var(--text-secondary)' }}>
          Combined Model Accuracy
        </div>
      </div>

      {/* Modal for Model Predictions */}
      {selectedModel && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1rem'
          }}
          onClick={closeModal}
        >
          <div 
            className="card"
            style={{
              width: '90%',
              maxWidth: '900px',
              maxHeight: '80vh',
              overflow: 'auto',
              position: 'relative'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div 
                  style={{
                    width: '2.5rem',
                    height: '2.5rem',
                    backgroundColor: 'var(--primary-100)',
                    borderRadius: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--primary-600)'
                  }}
                >
                  <selectedModel.icon size={20} />
                </div>
                <div>
                  <h2 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
                    {selectedModel.name} Predictions
                  </h2>
                  <p style={{ color: 'var(--text-secondary)' }}>
                    {selectedModel.description} - {selectedModel.accuracy} Accuracy
                  </p>
                </div>
              </div>
              <button
                onClick={closeModal}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  padding: '0.5rem',
                  borderRadius: '0.25rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <X size={24} />
              </button>
            </div>

            {/* Accuracy Stats */}
            <div className="grid-cols-3" style={{ marginBottom: '2rem' }}>
              <div className="card" style={{ textAlign: 'center', backgroundColor: 'var(--background-secondary)' }}>
                <div className="text-2xl font-bold" style={{ color: 'var(--primary-600)' }}>
                  {selectedModel.accuracy}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  Overall Accuracy
                </div>
              </div>
              <div className="card" style={{ textAlign: 'center', backgroundColor: 'var(--background-secondary)' }}>
                <div className="text-2xl font-bold" style={{ color: 'var(--accent-500)' }}>
                  {(Math.random() * 5 + 2).toFixed(1)}%
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  Mean Error
                </div>
              </div>
              <div className="card" style={{ textAlign: 'center', backgroundColor: 'var(--background-secondary)' }}>
                <div className="text-2xl font-bold" style={{ color: 'var(--success-500)' }}>
                  {(Math.random() * 0.2 + 0.8).toFixed(2)}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  R² Score
                </div>
              </div>
            </div>

            {/* Nivo Line Chart */}
            <div className="card" style={{ backgroundColor: 'var(--background-secondary)' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <TrendingUp size={20} />
                Prediction vs Actual Values (Last 20 Days)
              </h3>
              
              <div style={{ height: '400px' }}>
                <ResponsiveLine
                  data={[
                    {
                      id: 'Predicted',
                      color: '#3b82f6',
                      data: selectedModel.predictions.map(point => ({
                        x: `Day ${point.day}`,
                        y: parseFloat(point.prediction)
                      }))
                    },
                    {
                      id: 'Actual',
                      color: '#f59e0b',
                      data: selectedModel.predictions.map(point => ({
                        x: `Day ${point.day}`,
                        y: parseFloat(point.actual)
                      }))
                    }
                  ]}
                  margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
                  xScale={{ type: 'point' }}
                  yScale={{
                    type: 'linear',
                    min: 'auto',
                    max: 'auto',
                    stacked: false,
                    reverse: false
                  }}
                  yFormat=" >-.2f"
                  axisTop={null}
                  axisRight={null}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: -45,
                    legend: 'Days',
                    legendOffset: 36,
                    legendPosition: 'middle'
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'Accuracy (%)',
                    legendOffset: -40,
                    legendPosition: 'middle'
                  }}
                  pointSize={8}
                  pointColor={{ theme: 'background' }}
                  pointBorderWidth={2}
                  pointBorderColor={{ from: 'serieColor' }}
                  pointLabelYOffset={-12}
                  useMesh={true}
                  enableSlices="x"
                  sliceTooltip={({ slice }) => (
                    <div
                      style={{
                        background: 'var(--background-primary)',
                        padding: '12px 16px',
                        border: '1px solid var(--border-color)',
                        borderRadius: '8px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                        color: 'var(--text-primary)'
                      }}
                    >
                      <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
                        {slice.points[0].data.x}
                      </div>
                      {slice.points.map((point) => (
                        <div
                          key={point.id}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '4px'
                          }}
                        >
                          <div
                            style={{
                              width: '12px',
                              height: '12px',
                              backgroundColor: point.serieColor,
                              borderRadius: '2px'
                            }}
                          />
                          <span style={{ fontWeight: '500' }}>{point.serieId}:</span>
                          <span>{point.data.yFormatted}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                  legends={[
                    {
                      anchor: 'bottom-right',
                      direction: 'column',
                      justify: false,
                      translateX: 100,
                      translateY: 0,
                      itemsSpacing: 0,
                      itemDirection: 'left-to-right',
                      itemWidth: 80,
                      itemHeight: 20,
                      itemOpacity: 0.75,
                      symbolSize: 12,
                      symbolShape: 'circle',
                      symbolBorderColor: 'rgba(0, 0, 0, .5)',
                      effects: [
                        {
                          on: 'hover',
                          style: {
                            itemBackground: 'rgba(0, 0, 0, .03)',
                            itemOpacity: 1
                          }
                        }
                      ]
                    }
                  ]}
                  theme={{
                    background: 'transparent',
                    text: {
                      fill: 'var(--text-secondary)',
                      fontSize: 12
                    },
                    axis: {
                      domain: {
                        line: {
                          stroke: 'var(--border-color)',
                          strokeWidth: 1
                        }
                      },
                      legend: {
                        text: {
                          fill: 'var(--text-primary)',
                          fontSize: 12,
                          fontWeight: 600
                        }
                      },
                      ticks: {
                        line: {
                          stroke: 'var(--border-color)',
                          strokeWidth: 1
                        },
                        text: {
                          fill: 'var(--text-secondary)',
                          fontSize: 11
                        }
                      }
                    },
                    grid: {
                      line: {
                        stroke: 'var(--border-color)',
                        strokeWidth: 0.5,
                        strokeOpacity: 0.5
                      }
                    },
                    crosshair: {
                      line: {
                        stroke: 'var(--primary-500)',
                        strokeWidth: 2,
                        strokeOpacity: 0.75
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Data Table */}
            <div className="card" style={{ backgroundColor: 'var(--background-secondary)', marginTop: '1rem' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                Recent Predictions
              </h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(4, 1fr)', 
                gap: '0.5rem',
                maxHeight: '200px',
                overflow: 'auto'
              }}>
                <div style={{ fontWeight: 'bold', color: 'var(--text-primary)', padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>Day</div>
                <div style={{ fontWeight: 'bold', color: 'var(--text-primary)', padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>Predicted</div>
                <div style={{ fontWeight: 'bold', color: 'var(--text-primary)', padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>Actual</div>
                <div style={{ fontWeight: 'bold', color: 'var(--text-primary)', padding: '0.5rem', borderBottom: '1px solid var(--border-color)' }}>Error</div>
                
                {selectedModel.predictions.slice(-8).map((point, index) => (
                  <React.Fragment key={index}>
                    <div style={{ color: 'var(--text-secondary)', padding: '0.5rem' }}>{point.day}</div>
                    <div style={{ color: 'var(--primary-600)', padding: '0.5rem' }}>{point.prediction}%</div>
                    <div style={{ color: 'var(--accent-500)', padding: '0.5rem' }}>{point.actual}%</div>
                    <div style={{ color: Math.abs(point.prediction - point.actual) > 5 ? 'var(--error-500)' : 'var(--success-500)', padding: '0.5rem' }}>
                      {Math.abs(point.prediction - point.actual).toFixed(1)}%
                    </div>
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelInsights;