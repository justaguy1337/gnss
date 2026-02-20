import { 
  FileText, 
  Users, 
  ExternalLink,
  GraduationCap,
  Award,
  BookOpen,
  TrendingUp
} from 'lucide-react';

const Research = () => {
  const publications = [
    {
      title: 'Deep Learning Approaches for GNSS Error Prediction',
      authors: 'Smith, J., Johnson, A., Brown, K.',
      journal: 'IEEE Transactions on Aerospace and Electronic Systems',
      year: 2024,
      citations: 127,
      type: 'Journal Article'
    },
    {
      title: 'Ensemble Methods for Satellite Clock Error Forecasting',
      authors: 'Chen, L., Wilson, M., Davis, R.',
      journal: 'Journal of Navigation',
      year: 2023,
      citations: 89,
      type: 'Conference Paper'
    },
    {
      title: 'Real-time GNSS Error Correction Using ML',
      authors: 'Taylor, S., Anderson, P.',
      journal: 'GPS Solutions',
      year: 2023,
      citations: 156,
      type: 'Journal Article'
    }
  ];

  const collaborations = [
    { name: 'MIT AeroAstro', type: 'Academic', focus: 'ML Algorithms' },
    { name: 'Stanford GPS Lab', type: 'Academic', focus: 'Signal Processing' },
    { name: 'ESA Navigation', type: 'Government', focus: 'Galileo Integration' },
    { name: 'NASA JPL', type: 'Government', focus: 'Deep Space Networks' }
  ];

  const metrics = [
    { label: 'Publications', value: '47', icon: FileText },
    { label: 'Citations', value: '1,234', icon: TrendingUp },
    { label: 'Collaborators', value: '23', icon: Users },
    { label: 'Patents', value: '8', icon: Award }
  ];

  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Research & Publications
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Advancing the science of GNSS error prediction through cutting-edge research
        </p>
      </div>

      {/* Research Metrics */}
      <div className="grid-cols-4" style={{ marginBottom: '2rem' }}>
        {metrics.map((metric, index) => (
          <div key={metric.label} className="card animate-slide-up" style={{ animationDelay: `${index * 100}ms`, textAlign: 'center' }}>
            <div 
              style={{
                width: '3rem',
                height: '3rem',
                backgroundColor: 'var(--primary-100)',
                borderRadius: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem',
                color: 'var(--primary-600)'
              }}
            >
              <metric.icon size={24} />
            </div>
            <div className="text-3xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
              {metric.value}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              {metric.label}
            </div>
          </div>
        ))}
      </div>

      {/* Recent Publications */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--text-primary)' }}>
          Recent Publications
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {publications.map((pub, index) => (
            <div 
              key={index} 
              className="animate-slide-up"
              style={{ 
                animationDelay: `${index * 150}ms`,
                padding: '1.5rem',
                backgroundColor: 'var(--surface-alt)',
                borderRadius: '0.75rem',
                borderLeft: '4px solid var(--primary-500)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '0.75rem' }}>
                <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)', flex: 1 }}>
                  {pub.title}
                </h3>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span 
                    style={{ 
                      backgroundColor: 'var(--accent-100)',
                      color: 'var(--accent-600)',
                      padding: '0.25rem 0.5rem',
                      borderRadius: '0.25rem',
                      fontSize: '0.75rem',
                      fontWeight: '500'
                    }}
                  >
                    {pub.type}
                  </span>
                  <ExternalLink size={16} style={{ color: 'var(--text-muted)' }} />
                </div>
              </div>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                {pub.authors}
              </p>
              <p style={{ color: 'var(--text-muted)', marginBottom: '0.75rem', fontStyle: 'italic' }}>
                {pub.journal} ({pub.year})
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  Citations: {pub.citations}
                </span>
                <div style={{ width: '4px', height: '4px', backgroundColor: 'var(--text-muted)', borderRadius: '50%' }}></div>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  Year: {pub.year}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Research Collaborations */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--text-primary)' }}>
          Research Collaborations
        </h2>
        <div className="grid-cols-2">
          {collaborations.map((collab, index) => (
            <div 
              key={collab.name} 
              className="animate-slide-up"
              style={{ 
                animationDelay: `${index * 150}ms`,
                padding: '1.5rem',
                backgroundColor: 'var(--surface-alt)',
                borderRadius: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
            >
              <div 
                style={{
                  width: '3rem',
                  height: '3rem',
                  backgroundColor: collab.type === 'Academic' ? 'var(--accent-100)' : 'var(--primary-100)',
                  borderRadius: '0.75rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: collab.type === 'Academic' ? 'var(--accent-600)' : 'var(--primary-600)'
                }}
              >
                {collab.type === 'Academic' ? <GraduationCap size={20} /> : <BookOpen size={20} />}
              </div>
              <div>
                <h3 className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {collab.name}
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  {collab.focus}
                </p>
                <span 
                  style={{ 
                    backgroundColor: collab.type === 'Academic' ? 'var(--accent-100)' : 'var(--primary-100)',
                    color: collab.type === 'Academic' ? 'var(--accent-600)' : 'var(--primary-600)',
                    padding: '0.125rem 0.5rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    fontWeight: '500'
                  }}
                >
                  {collab.type}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Research Focus Areas */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--text-primary)' }}>
          Current Research Focus
        </h2>
        <div className="grid-cols-3">
          <div style={{ textAlign: 'center' }}>
            <div 
              style={{
                width: '4rem',
                height: '4rem',
                backgroundColor: 'var(--primary-100)',
                borderRadius: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem',
                color: 'var(--primary-600)'
              }}
            >
              <FileText size={32} />
            </div>
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              Machine Learning
            </h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Advanced neural networks for pattern recognition in satellite data
            </p>
          </div>
          <div style={{ textAlign: 'center' }}>
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
              <TrendingUp size={32} />
            </div>
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              Signal Processing
            </h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Real-time analysis of GNSS signals for error detection
            </p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div 
              style={{
                width: '4rem',
                height: '4rem',
                backgroundColor: 'var(--primary-100)',
                borderRadius: '1rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem',
                color: 'var(--primary-600)'
              }}
            >
              <Award size={32} />
            </div>
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              System Integration
            </h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Seamless integration with existing GNSS infrastructure
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Research;