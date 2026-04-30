import React, { useRef, useEffect, useState } from 'react';
import Globe from 'react-globe.gl';
import { Satellite, Globe as GlobeIcon, Play, Pause, RotateCcw, Settings } from 'lucide-react';

const EarthVisualization = () => {
  const globeEl = useRef();
  const [isRotating, setIsRotating] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(0.5);
  const [satelliteData, setSatelliteData] = useState([]);
  const [time, setTime] = useState(0);

  // Generate satellite orbit data
  useEffect(() => {
    const satellites = [
      {
        id: 1,
        name: 'GPS Satellite 1',
        altitude: 20200,
        inclination: 55,
        speed: 0.01, // Reduced speed for smoother movement
        color: '#ff6b6b',
        size: 0.8
      },
      {
        id: 2,
        name: 'GPS Satellite 2',
        altitude: 20200,
        inclination: 55,
        speed: 0.009, // Reduced speed for smoother movement
        color: '#4ecdc4',
        size: 0.8
      },
      {
        id: 3,
        name: 'GPS Satellite 3',
        altitude: 20200,
        inclination: 55,
        speed: 0.011, // Reduced speed for smoother movement
        color: '#45b7d1',
        size: 0.8
      },
      {
        id: 4,
        name: 'ISS',
        altitude: 408,
        inclination: 51.6,
        speed: 0.03, // Reduced speed for smoother movement
        color: '#96ceb4',
        size: 1.2
      }
    ];

    const generateSatellitePositions = (time) => {
      return satellites.map((sat, index) => {
        // Add phase offset for different satellites to spread them out
        const phaseOffset = (index * Math.PI * 2) / satellites.length;
        const angle = (time * sat.speed + phaseOffset) % (2 * Math.PI);
        
        // Smoother orbital calculations with easing
        const smoothAngle = angle + Math.sin(angle * 2) * 0.1; // Add slight orbital perturbation
        
        // Convert to spherical coordinates with smoother transitions
        const lat = Math.sin(sat.inclination * Math.PI / 180) * Math.sin(smoothAngle) * 85; // Reduced max latitude for smoother movement
        const lng = (smoothAngle * 180 / Math.PI) % 360 - 180;
        const alt = sat.altitude / 6371; // Normalize altitude relative to Earth radius
        
        return {
          ...sat,
          lat,
          lng,
          alt
        };
      });
    };

    setSatelliteData(generateSatellitePositions(time));
  }, [time]);

  // Animation loop with smoother intervals
  useEffect(() => {
    if (!isRotating) return;
    
    const interval = setInterval(() => {
      setTime(prevTime => prevTime + 0.5); // Smaller time increments for smoother movement
      
      // Smoother auto-rotate the globe
      if (globeEl.current) {
        const currentPOV = globeEl.current.pointOfView();
        globeEl.current.pointOfView({
          lng: (currentPOV.lng + rotationSpeed * 0.5) % 360 // Smaller rotation increments
        }, 50); // Shorter transition time for smoother rotation
      }
    }, 50); // Faster update rate (20 FPS instead of 10 FPS)

    return () => clearInterval(interval);
  }, [isRotating, rotationSpeed]);

  const toggleRotation = () => {
    setIsRotating(!isRotating);
  };

  const resetView = () => {
    if (globeEl.current) {
      globeEl.current.pointOfView({
        lat: 0,
        lng: 0,
        altitude: 2.5
      }, 1000);
    }
  };

  const handleSpeedChange = (newSpeed) => {
    setRotationSpeed(newSpeed);
  };

  return (
    <div className="animate-fade-in" style={{ height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ 
        position: 'absolute', 
        top: '1rem', 
        left: '1rem', 
        right: '1rem', 
        zIndex: 1000,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div className="card" style={{
          padding: '0.75rem 1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)'
        }}>
          <GlobeIcon size={24} style={{ color: '#3b82f6' }} />
          <h2 style={{ 
            margin: 0, 
            fontSize: '1.25rem', 
            fontWeight: 'bold',
            color: 'var(--text-primary)'
          }}>
            Earth Visualization
          </h2>
        </div>

        {/* Controls */}
        <div className="card" style={{ 
          padding: '1rem', 
          display: 'flex', 
          gap: '0.5rem', 
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)'
        }}>
          <button
            onClick={toggleRotation}
            className="btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              border: 'none',
              backgroundColor: isRotating ? '#ef4444' : '#22c55e',
              color: 'white',
              cursor: 'pointer',
              fontSize: '0.875rem'
            }}
          >
            {isRotating ? <Pause size={16} /> : <Play size={16} />}
            {isRotating ? 'Pause' : 'Play'}
          </button>

          <button
            onClick={resetView}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              border: '1px solid var(--border-color)',
              backgroundColor: 'transparent',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '0.875rem'
            }}
          >
            <RotateCcw size={16} />
            Reset View
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings size={16} style={{ color: 'var(--text-secondary)' }} />
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Speed:</span>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.1"
              value={rotationSpeed}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              style={{ width: '80px' }}
            />
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', minWidth: '30px' }}>
              {rotationSpeed}x
            </span>
          </div>
        </div>
      </div>

      {/* Satellite Info Panel */}
      <div style={{
        position: 'absolute',
        bottom: '1rem',
        left: '1rem',
        zIndex: 1000,
        maxWidth: '300px'
      }}>
        <div className="card" style={{
          padding: '1rem',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)'
        }}>
          <h3 className="font-semibold mb-3" style={{ 
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <Satellite size={18} />
            Active Satellites
          </h3>
          
          {satelliteData.map(sat => (
            <div key={sat.id} style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              marginBottom: '0.5rem',
              fontSize: '0.875rem'
            }}>
              <div style={{
                width: '12px',
                height: '12px',
                backgroundColor: sat.color,
                borderRadius: '50%'
              }} />
              <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>
                {sat.name}
              </span>
              <span style={{ color: 'var(--text-secondary)', marginLeft: 'auto' }}>
                {sat.altitude} km
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Globe */}
      <Globe
        ref={globeEl}
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
        
        // Smooth rendering settings
        rendererConfig={{ 
          antialias: true,
          alpha: true,
          powerPreference: "high-performance"
        }}
        
        // Satellites as simple points
        pointsData={satelliteData}
        pointLat={d => d.lat}
        pointLng={d => d.lng}
        pointAltitude={d => d.alt}
        pointColor={d => d.color}
        pointRadius={d => d.size}
        pointResolution={8} // Smoother point rendering
        pointLabel={d => `
          <div style="background: rgba(0,0,0,0.8); color: white; padding: 8px; border-radius: 4px; font-size: 12px;">
            <div style="font-weight: bold; margin-bottom: 4px;">${d.name}</div>
            <div>Altitude: ${d.altitude} km</div>
            <div>Inclination: ${d.inclination}°</div>
          </div>
        `}
        
        // Smoother atmosphere
        showAtmosphere={true}
        atmosphereColor="#69b7ff"
        atmosphereAltitude={0.12}
        
        // Initial view
        pointOfView={{ lat: 0, lng: 0, altitude: 2.5 }}
        
        // Enhanced controls for smoother interaction
        enablePointerInteraction={true}
        pointerEventsFilter={() => true}
        
        // Smooth animation
        animateIn={true}
        
        // Better performance settings
        waitForGlobeReady={true}
      />
    </div>
  );
};

export default EarthVisualization;