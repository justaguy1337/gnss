import { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  Satellite,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';

const Dashboard = () => {
  const [realTimeData, setRealTimeData] = useState([]);
  const [isLive, setIsLive] = useState(true);

  // Generate dummy real-time data
  useEffect(() => {
    const generateData = () => {
      const now = new Date();
      const data = [];
      
      for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        data.push({
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          clockError: Math.random() * 10 + Math.sin(i / 4) * 5,
          ephemerisError: Math.random() * 8 + Math.cos(i / 3) * 4,
          prediction: Math.random() * 9 + Math.sin(i / 5) * 3,
          uncertainty: Math.random() * 2 + 1,
        });
      }
      return data;
    };

    setRealTimeData(generateData());

    if (isLive) {
      const interval = setInterval(() => {
        setRealTimeData(prev => {
          const newData = [...prev.slice(1)];
          const now = new Date();
          newData.push({
            time: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
            clockError: Math.random() * 10 + Math.sin(Date.now() / 10000) * 5,
            ephemerisError: Math.random() * 8 + Math.cos(Date.now() / 8000) * 4,
            prediction: Math.random() * 9 + Math.sin(Date.now() / 12000) * 3,
            uncertainty: Math.random() * 2 + 1,
          });
          return newData;
        });
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [isLive]);

  const metrics = [
    {
      title: 'Active Satellites',
      value: '32',
      change: '+2',
      changeType: 'positive',
      icon: Satellite,
    },
    {
      title: 'Prediction Accuracy',
      value: '94.2%',
      change: '+1.2%',
      changeType: 'positive',
      icon: TrendingUp,
    },
    {
      title: 'Avg Response Time',
      value: '1.2ms',
      change: '-0.3ms',
      changeType: 'positive',
      icon: Clock,
    },
    {
      title: 'System Status',
      value: 'Operational',
      change: 'All systems normal',
      changeType: 'positive',
      icon: CheckCircle2,
    }
  ];

  return (
    <div className="dashboard animate-fade-in">
      {/* Header */}
      <div className="dashboard-header">
        <div>
          <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
            GNSS ErrorNet Dashboard
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Real-time satellite error prediction and monitoring
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div 
              style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                backgroundColor: isLive ? '#10B981' : '#6B7280',
                animation: isLive ? 'pulse 2s infinite' : 'none'
              }}
            ></div>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: '500' }}>
              {isLive ? 'Live Data' : 'Static Data'}
            </span>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            className="btn-primary"
          >
            {isLive ? 'Pause' : 'Resume'}
          </button>
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
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  color: metric.changeType === 'positive' ? '#10B981' : '#EF4444'
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

      {/* Real-time Charts */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {/* Error Trends Chart */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              Real-time Error Trends
            </h2>
            <Activity size={20} style={{ color: 'var(--text-muted)' }} />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={realTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="clockError" 
                stroke="#EF4444" 
                strokeWidth={2}
                name="Clock Error (ns)"
              />
              <Line 
                type="monotone" 
                dataKey="ephemerisError" 
                stroke="#F59E0B" 
                strokeWidth={2}
                name="Ephemeris Error (m)"
              />
              <Line 
                type="monotone" 
                dataKey="prediction" 
                stroke="#10B981" 
                strokeWidth={2}
                name="ML Prediction"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Uncertainty Intervals */}
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h2 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              Prediction Uncertainty
            </h2>
            <AlertTriangle size={20} style={{ color: '#F59E0B' }} />
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={realTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="time" stroke="#6B7280" fontSize={12} />
              <YAxis stroke="#6B7280" fontSize={12} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'var(--bg-primary)',
                  border: 'none',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="uncertainty" 
                stroke="#8B5CF6" 
                fill="#8B5CF6" 
                fillOpacity={0.6}
                name="Uncertainty (±σ)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Simple status message */}
      <div className="card" style={{ textAlign: 'center' }}>
        <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          System Status
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
          <CheckCircle2 size={24} style={{ color: '#10B981' }} />
          <span style={{ color: 'var(--text-primary)', fontSize: '1.125rem', fontWeight: '500' }}>
            All systems operational
          </span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;