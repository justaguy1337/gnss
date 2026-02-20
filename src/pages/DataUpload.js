import { useState } from 'react';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  Clock
} from 'lucide-react';

const DataUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const newFiles = Array.from(e.dataTransfer.files).map(file => ({
        name: file.name,
        size: (file.size / 1024 / 1024).toFixed(2),
        status: 'uploading',
        progress: 0
      }));
      setFiles(prev => [...prev, ...newFiles]);
      
      // Simulate upload progress
      newFiles.forEach((file, index) => {
        const interval = setInterval(() => {
          setFiles(prev => prev.map(f => 
            f.name === file.name ? 
            { ...f, progress: Math.min(f.progress + 10, 100), status: f.progress >= 90 ? 'completed' : 'uploading' } 
            : f
          ));
        }, 200);
        
        setTimeout(() => clearInterval(interval), 2000);
      });
    }
  };

  const uploadStats = [
    { label: 'Files Processed', value: '1,247', icon: FileText },
    { label: 'Total Size', value: '15.2 GB', icon: Upload },
    { label: 'Success Rate', value: '99.8%', icon: CheckCircle },
    { label: 'Avg Processing Time', value: '2.3s', icon: Clock }
  ];

  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Data Upload & Processing
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Upload satellite data files for error prediction analysis
        </p>
      </div>

      {/* Upload Stats */}
      <div className="grid-cols-4" style={{ marginBottom: '2rem' }}>
        {uploadStats.map((stat, index) => (
          <div key={stat.label} className="card animate-slide-up" style={{ animationDelay: `${index * 100}ms`, textAlign: 'center' }}>
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
              <stat.icon size={24} />
            </div>
            <div className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
              {stat.value}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Upload Area */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 className="text-2xl font-semibold mb-6" style={{ color: 'var(--text-primary)' }}>
          Upload Satellite Data
        </h2>
        
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragActive ? 'var(--primary-500)' : 'var(--border-color)'}`,
            borderRadius: '0.75rem',
            padding: '3rem 2rem',
            textAlign: 'center',
            backgroundColor: dragActive ? 'var(--primary-50)' : 'var(--surface-alt)',
            transition: 'all 0.3s ease',
            cursor: 'pointer'
          }}
        >
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
            <Upload size={32} />
          </div>
          <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Drag & Drop Files Here
          </h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Or click to browse files. Supports RINEX, SP3, CLK formats
          </p>
          <button 
            style={{
              backgroundColor: 'var(--primary-500)',
              color: 'white',
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              fontWeight: '500',
              cursor: 'pointer'
            }}
          >
            Select Files
          </button>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Upload Queue
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {files.map((file, index) => (
              <div 
                key={index}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '1rem',
                  backgroundColor: 'var(--surface-alt)',
                  borderRadius: '0.5rem',
                  gap: '1rem'
                }}
              >
                <div 
                  style={{
                    width: '2.5rem',
                    height: '2.5rem',
                    backgroundColor: file.status === 'completed' ? 'var(--green-100)' : 'var(--primary-100)',
                    borderRadius: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: file.status === 'completed' ? 'var(--green-600)' : 'var(--primary-600)'
                  }}
                >
                  {file.status === 'completed' ? <CheckCircle size={16} /> : <FileText size={16} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                    <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>{file.name}</span>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>{file.size} MB</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div 
                      style={{
                        flex: 1,
                        height: '4px',
                        backgroundColor: 'var(--surface)',
                        borderRadius: '2px',
                        overflow: 'hidden'
                      }}
                    >
                      <div 
                        style={{
                          height: '100%',
                          backgroundColor: file.status === 'completed' ? 'var(--green-500)' : 'var(--primary-500)',
                          width: `${file.progress}%`,
                          transition: 'width 0.3s ease'
                        }}
                      ></div>
                    </div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                      {file.progress}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataUpload;