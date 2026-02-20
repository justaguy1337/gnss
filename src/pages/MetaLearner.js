import React, { useState } from 'react';
import './MetaLearner.css';

const MetaLearner = () => {
  const [showResults, setShowResults] = useState(false);
  const [isComputing, setIsComputing] = useState(false);
  const [showAnimations, setShowAnimations] = useState(false);

  const models = [
    { name: 'LSTM-GRU', accuracy: 91.2, color: '#4285f4', position: { top: '20%', left: '25%' } },
    { name: 'TimeGPT', accuracy: 94.7, color: '#9c27b0', position: { top: '20%', left: '70%' } },
    { name: 'XGBoost', accuracy: 89.5, color: '#4caf50', position: { top: '50%', left: '85%' } },
    { name: 'Gaussian Process', accuracy: 87.3, color: '#ff9800', position: { top: '80%', left: '70%' } },
    { name: 'GAN', accuracy: 92.8, color: '#f44336', position: { top: '80%', left: '25%' } },
    { name: 'TCN', accuracy: 90.2, color: '#00bcd4', position: { top: '50%', left: '10%' } }
  ];

  const finalAccuracy = 93.3;

  const handleCompute = () => {
    setIsComputing(true);
    setShowAnimations(true);
    
    // Run animations for 10 seconds, then show results
    setTimeout(() => {
      setIsComputing(false);
      setShowResults(true);
      setTimeout(() => {
        document.getElementById('results-section').scrollIntoView({ 
          behavior: 'smooth' 
        });
      }, 300);
    }, 10000); // 10 seconds
  };

  const resetVisualization = () => {
    setShowResults(false);
    setIsComputing(false);
    setShowAnimations(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="meta-learner-simple">
      {/* Header */}
      <div className="simple-header">
        <h1>Meta Learner Ensemble</h1>
        <p>Combining multiple models for superior GNSS error prediction accuracy</p>
      </div>

      {/* Simple Mindmap */}
      <div className="simple-mindmap">
        {/* SVG for proper connections - only show animations when computing */}
        <svg className="connections-svg" viewBox="0 0 1000 600">
          {/* Connection lines from each model to center */}
          <line x1="250" y1="120" x2="500" y2="300" stroke="#4285f4" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '0s'}} />
          <line x1="700" y1="120" x2="500" y2="300" stroke="#9c27b0" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '0.3s'}} />
          <line x1="800" y1="300" x2="500" y2="300" stroke="#4caf50" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '0.6s'}} />
          <line x1="750" y1="480" x2="500" y2="300" stroke="#ff9800" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '0.9s'}} />
          <line x1="250" y1="480" x2="500" y2="300" stroke="#f44336" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '1.2s'}} />
          <line x1="100" y1="300" x2="500" y2="300" stroke="#00bcd4" strokeWidth="2" 
                className={`connection-svg-line ${showAnimations ? 'animated' : ''}`} 
                style={{animationDelay: '1.5s'}} />
          
          {/* Animated particles - only show when computing */}
          {showAnimations && (
            <>
              <circle r="4" fill="#4285f4" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="0s">
                  <mpath href="#path1"/>
                </animateMotion>
              </circle>
              <circle r="4" fill="#9c27b0" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="0.3s">
                  <mpath href="#path2"/>
                </animateMotion>
              </circle>
              <circle r="4" fill="#4caf50" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="0.6s">
                  <mpath href="#path3"/>
                </animateMotion>
              </circle>
              <circle r="4" fill="#ff9800" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="0.9s">
                  <mpath href="#path4"/>
                </animateMotion>
              </circle>
              <circle r="4" fill="#f44336" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="1.2s">
                  <mpath href="#path5"/>
                </animateMotion>
              </circle>
              <circle r="4" fill="#00bcd4" className="connection-particle">
                <animateMotion dur="2s" repeatCount="indefinite" begin="1.5s">
                  <mpath href="#path6"/>
                </animateMotion>
              </circle>
            </>
          )}
          
          {/* Hidden paths for animation */}
          <defs>
            <path id="path1" d="M 250 120 L 500 300" />
            <path id="path2" d="M 700 120 L 500 300" />
            <path id="path3" d="M 800 300 L 500 300" />
            <path id="path4" d="M 750 480 L 500 300" />
            <path id="path5" d="M 250 480 L 500 300" />
            <path id="path6" d="M 100 300 L 500 300" />
          </defs>
        </svg>

        {/* Central Meta Learner */}
        <div className="meta-center">
          <div className={`center-circle ${isComputing ? 'computing' : ''}`}>
            <h3>Meta Learner</h3>
            <p>Ensemble</p>
            {showResults && (
              <div className="final-result">
                <strong>{finalAccuracy}%</strong>
              </div>
            )}
            {isComputing && (
              <div className="computing-indicator">
                <div className="loading-spinner"></div>
                <span>Computing...</span>
              </div>
            )}
          </div>
        </div>

        {/* Model Nodes */}
        {models.map((model, index) => (
          <div
            key={model.name}
            className="model-circle"
            style={{
              ...model.position,
              borderColor: model.color
            }}
          >
            <h4>{model.name}</h4>
            <span>{model.accuracy}%</span>
          </div>
        ))}

        {/* Compute Button */}
        {!isComputing && !showResults && (
          <button className="simple-compute-btn" onClick={handleCompute}>
            Compute Final Accuracy
          </button>
        )}

        {/* Reset Button */}
        {(showResults || isComputing) && (
          <button className="simple-reset-btn" onClick={resetVisualization}>
            Reset
          </button>
        )}
      </div>

      {/* Simple Results Section */}
      {showResults && (
        <div id="results-section" className="simple-results">
          <h2>Results</h2>
          
          <div className="results-cards">
            {/* Individual Models */}
            <div className="result-card">
              <h3>Individual Model Performance</h3>
              {models.map((model) => (
                <div key={model.name} className="model-result">
                  <span className="model-name">{model.name}</span>
                  <div className="accuracy-bar">
                    <div 
                      className="accuracy-fill" 
                      style={{ 
                        width: `${model.accuracy}%`,
                        backgroundColor: model.color
                      }}
                    ></div>
                  </div>
                  <span className="accuracy-text">{model.accuracy}%</span>
                </div>
              ))}
            </div>

            {/* Ensemble Results */}
            <div className="result-card">
              <h3>Ensemble Results</h3>
              <div className="final-metrics">
                <div className="metric">
                  <strong>{finalAccuracy}%</strong>
                  <span>Final Accuracy</span>
                </div>
                <div className="metric">
                  <strong>+{(finalAccuracy - Math.max(...models.map(m => m.accuracy))).toFixed(1)}%</strong>
                  <span>Improvement</span>
                </div>
                <div className="metric">
                  <strong>6</strong>
                  <span>Models Combined</span>
                </div>
              </div>
            </div>

            {/* Line Graph Chart */}
            <div className="result-card full-width">
              <h3>Accuracy Comparison</h3>
              <div className="line-chart-container">
                <svg className="line-chart" viewBox="0 0 800 300">
                  {/* Grid lines */}
                  <defs>
                    <pattern id="grid" width="80" height="30" patternUnits="userSpaceOnUse">
                      <path d="M 80 0 L 0 0 0 30" fill="none" stroke="#e0e0e0" strokeWidth="1"/>
                    </pattern>
                  </defs>
                  <rect width="800" height="300" fill="url(#grid)" />
                  
                  {/* Y-axis labels */}
                  <text x="30" y="280" textAnchor="middle" className="chart-axis-label">85%</text>
                  <text x="30" y="220" textAnchor="middle" className="chart-axis-label">88%</text>
                  <text x="30" y="160" textAnchor="middle" className="chart-axis-label">91%</text>
                  <text x="30" y="100" textAnchor="middle" className="chart-axis-label">94%</text>
                  <text x="30" y="40" textAnchor="middle" className="chart-axis-label">97%</text>
                  
                  {/* Line path */}
                  <path
                    d="M 80 140 L 180 80 L 280 160 L 380 200 L 480 120 L 580 150 L 680 130"
                    fill="none"
                    stroke="#4caf50"
                    strokeWidth="3"
                    className="line-path"
                  />
                  
                  {/* Data points */}
                  <circle cx="80" cy="140" r="6" fill="#4285f4" className="data-point" />
                  <circle cx="180" cy="80" r="6" fill="#9c27b0" className="data-point" />
                  <circle cx="280" cy="160" r="6" fill="#4caf50" className="data-point" />
                  <circle cx="380" cy="200" r="6" fill="#ff9800" className="data-point" />
                  <circle cx="480" cy="120" r="6" fill="#f44336" className="data-point" />
                  <circle cx="580" cy="150" r="6" fill="#00bcd4" className="data-point" />
                  <circle cx="680" cy="130" r="8" fill="#333" className="data-point meta-point" />
                  
                  {/* X-axis labels */}
                  <text x="80" y="295" textAnchor="middle" className="chart-label">LSTM-GRU</text>
                  <text x="180" y="295" textAnchor="middle" className="chart-label">TimeGPT</text>
                  <text x="280" y="295" textAnchor="middle" className="chart-label">XGBoost</text>
                  <text x="380" y="295" textAnchor="middle" className="chart-label">Gaussian</text>
                  <text x="480" y="295" textAnchor="middle" className="chart-label">GAN</text>
                  <text x="580" y="295" textAnchor="middle" className="chart-label">TCN</text>
                  <text x="680" y="295" textAnchor="middle" className="chart-label meta-label">Meta Learner</text>
                  
                  {/* Accuracy values on hover */}
                  <text x="80" y="130" textAnchor="middle" className="accuracy-value-text">91.2%</text>
                  <text x="180" y="70" textAnchor="middle" className="accuracy-value-text">94.7%</text>
                  <text x="280" y="150" textAnchor="middle" className="accuracy-value-text">89.5%</text>
                  <text x="380" y="190" textAnchor="middle" className="accuracy-value-text">87.3%</text>
                  <text x="480" y="110" textAnchor="middle" className="accuracy-value-text">92.8%</text>
                  <text x="580" y="140" textAnchor="middle" className="accuracy-value-text">90.2%</text>
                  <text x="680" y="120" textAnchor="middle" className="accuracy-value-text meta-value">93.3%</text>
                </svg>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MetaLearner;