import { 
  Satellite, 
  Navigation, 
  Shield, 
  DollarSign, 
  Clock,
  Users,
  Globe
} from 'lucide-react';

const ImpactBenefits = () => {
  const benefits = [
    {
      icon: Shield,
      title: 'Enhanced Safety',
      description: 'Prevents navigation errors that could affect aviation and maritime safety',
      metric: '99.2%',
      metricLabel: 'Error Prevention'
    },
    {
      icon: DollarSign,
      title: 'Cost Savings',
      description: 'Reduces operational costs by minimizing satellite downtime',
      metric: '$15M',
      metricLabel: 'Annual Savings'
    },
    {
      icon: Clock,
      title: 'Faster Response',
      description: 'Real-time predictions enable proactive maintenance',
      metric: '2.3x',
      metricLabel: 'Faster Detection'
    },
    {
      icon: Users,
      title: 'Improved Reliability',
      description: 'Higher system availability for critical applications',
      metric: '99.9%',
      metricLabel: 'System Uptime'
    }
  ];

  const applications = [
    { name: 'Aviation Navigation', users: '2.8M flights/year', icon: Navigation },
    { name: 'Maritime Systems', users: '145K vessels', icon: Globe },
    { name: 'Emergency Services', users: '98% coverage', icon: Shield },
    { name: 'Precision Agriculture', users: '560K farms', icon: Satellite }
  ];

  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Impact & Benefits
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Transforming satellite navigation through predictive error correction
        </p>
      </div>

      {/* Key Benefits */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {benefits.map((benefit, index) => (
          <div key={benefit.title} className="card animate-slide-up" style={{ animationDelay: `${index * 100}ms` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
              <div 
                style={{
                  width: '3rem',
                  height: '3rem',
                  backgroundColor: 'var(--primary-100)',
                  borderRadius: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--primary-600)'
                }}
              >
                <benefit.icon size={24} />
              </div>
              <div>
                <h3 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {benefit.title}
                </h3>
                <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {benefit.description}
                </p>
              </div>
            </div>
            <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'var(--surface-alt)', borderRadius: '0.5rem' }}>
              <div className="text-3xl font-bold" style={{ color: 'var(--accent-500)' }}>
                {benefit.metric}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                {benefit.metricLabel}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Applications */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--text-primary)' }}>
          Critical Applications
        </h2>
        <div className="grid-cols-2">
          {applications.map((app, index) => (
            <div 
              key={app.name} 
              className="animate-slide-up"
              style={{ 
                animationDelay: `${index * 150}ms`,
                padding: '1.5rem',
                backgroundColor: 'var(--surface-alt)',
                borderRadius: '0.75rem',
                textAlign: 'center'
              }}
            >
              <div 
                style={{
                  width: '4rem',
                  height: '4rem',
                  backgroundColor: 'var(--accent-100)',
                  borderRadius: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1rem',
                  color: 'var(--accent-600)'
                }}
              >
                <app.icon size={32} />
              </div>
              <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                {app.name}
              </h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                {app.users}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Overall Impact */}
      <div className="card" style={{ textAlign: 'center', background: 'linear-gradient(135deg, var(--primary-500), var(--accent-500))' }}>
        <div style={{ color: 'white' }}>
          <h2 className="text-3xl font-bold mb-4">
            Global Impact
          </h2>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '3rem', flexWrap: 'wrap' }}>
            <div>
              <div className="text-4xl font-bold mb-2">
                3.2B
              </div>
              <div style={{ opacity: 0.9 }}>
                People Served
              </div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">
                $2.8B
              </div>
              <div style={{ opacity: 0.9 }}>
                Economic Value
              </div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">
                125
              </div>
              <div style={{ opacity: 0.9 }}>
                Countries
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImpactBenefits;