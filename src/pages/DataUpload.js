import { useState, useRef } from 'react';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  Clock,
  Play,
  Database,
  AlertCircle,
  Loader2
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const DataUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadResult, setUploadResult] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [generatingData, setGeneratingData] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadResult(data);
        return { success: true, data };
      } else {
        const err = await res.json();
        return { success: false, error: err.detail };
      }
    } catch (e) {
      return { success: false, error: 'API server not running. Start with: python ml/api.py' };
    }
  };

  const handleFiles = async (fileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      name: file.name,
      size: (file.size / 1024 / 1024).toFixed(2),
      status: 'uploading',
      progress: 0,
      raw: file
    }));
    setFiles(prev => [...prev, ...newFiles]);

    for (const fileEntry of newFiles) {
      // Animate progress
      const progressInterval = setInterval(() => {
        setFiles(prev => prev.map(f =>
          f.name === fileEntry.name && f.status === 'uploading'
            ? { ...f, progress: Math.min(f.progress + 15, 85) }
            : f
        ));
      }, 200);

      const result = await uploadFile(fileEntry.raw);

      clearInterval(progressInterval);
      setFiles(prev => prev.map(f =>
        f.name === fileEntry.name
          ? { ...f, progress: 100, status: result.success ? 'completed' : 'error', error: result.error }
          : f
      ));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  };

  const generateSyntheticData = async () => {
    setGeneratingData(true);
    try {
      const res = await fetch(`${API_BASE}/generate-synthetic`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setUploadResult(data);
        setFiles([{
          name: 'gnss_errors_synthetic.csv',
          size: 'generated',
          status: 'completed',
          progress: 100
        }]);
      } else {
        setUploadResult({ status: 'error', message: 'Failed to generate synthetic data' });
      }
    } catch {
      setUploadResult({ status: 'error', message: 'API server not running. Start with: python ml/api.py' });
    } finally {
      setGeneratingData(false);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setTrainingStatus({ status: 'starting', message: 'Initializing training...' });

    try {
      const res = await fetch(`${API_BASE}/train?quick=true`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setTrainingStatus({ status: 'training', message: data.message });

        // Poll for status
        const pollInterval = setInterval(async () => {
          try {
            const statusRes = await fetch(`${API_BASE}/status`);
            if (statusRes.ok) {
              const status = await statusRes.json();
              setTrainingStatus({
                status: status.status,
                message: status.message,
                progress: status.progress
              });

              if (status.status === 'trained' || status.status === 'error') {
                clearInterval(pollInterval);
                setIsTraining(false);
              }
            }
          } catch { /* ignore polling errors */ }
        }, 3000);
      } else {
        const err = await res.json();
        setTrainingStatus({ status: 'error', message: err.detail });
        setIsTraining(false);
      }
    } catch {
      setTrainingStatus({ status: 'error', message: 'API server not running' });
      setIsTraining(false);
    }
  };

  const uploadStats = [
    { label: 'Required Format', value: 'CSV', icon: FileText },
    { label: 'Sample Rate', value: '15 min', icon: Clock },
    { label: 'Train Duration', value: '7 days', icon: Database },
    { label: 'Test Duration', value: '1 day', icon: CheckCircle }
  ];

  return (
    <div className="animate-fade-in">
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Data Upload & Processing
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Upload GNSS error data (CSV) or generate synthetic data for testing
        </p>
      </div>

      {/* Data Requirements */}
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

      {/* Upload Area + Synthetic Generator */}
      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>
        {/* Upload CSV */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Upload CSV Data
          </h2>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragActive ? 'var(--primary-500)' : 'var(--border-color)'}`,
              borderRadius: '0.75rem',
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              backgroundColor: dragActive ? 'var(--primary-50)' : 'var(--surface-alt)',
              transition: 'all 0.3s ease',
              cursor: 'pointer'
            }}
          >
            <Upload size={32} style={{ color: 'var(--primary-500)', margin: '0 auto 0.75rem', display: 'block' }} />
            <h3 className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
              Drag & Drop CSV Here
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
              Columns: timestamp, satellite_id, satellite_type, clock_error_ns, radial_error_m, along_track_error_m, cross_track_error_m
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>
        </div>

        {/* Generate Synthetic */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Synthetic Data
          </h2>
          <div style={{
            border: '2px dashed var(--border-color)',
            borderRadius: '0.75rem',
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            backgroundColor: 'var(--surface-alt)',
          }}>
            <Database size={32} style={{ color: '#8B5CF6', margin: '0 auto 0.75rem', display: 'block' }} />
            <h3 className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
              Generate Test Data
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
              Creates 8-day synthetic GNSS error data (4 MEO + 2 GEO satellites)
            </p>
            <button
              onClick={generateSyntheticData}
              disabled={generatingData}
              style={{
                backgroundColor: '#8B5CF6',
                color: 'white',
                padding: '0.6rem 1.25rem',
                borderRadius: '0.5rem',
                border: 'none',
                fontWeight: '500',
                cursor: generatingData ? 'not-allowed' : 'pointer',
                opacity: generatingData ? 0.6 : 1,
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                margin: '0 auto'
              }}
            >
              {generatingData ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Database size={16} />}
              {generatingData ? 'Generating...' : 'Generate Synthetic Data'}
            </button>
          </div>
        </div>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="card" style={{ marginBottom: '2rem', borderLeft: `4px solid ${uploadResult.status === 'success' ? '#10B981' : '#EF4444'}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            {uploadResult.status === 'success' ? (
              <CheckCircle size={20} style={{ color: '#10B981' }} />
            ) : (
              <AlertCircle size={20} style={{ color: '#EF4444' }} />
            )}
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              {uploadResult.message}
            </span>
          </div>
          {uploadResult.summary && (
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginLeft: '2rem' }}>
              <p>{uploadResult.summary.n_satellites} satellites | {uploadResult.summary.total_rows} rows</p>
              <p>{uploadResult.summary.n_meo} MEO + {uploadResult.summary.n_geo} GEO</p>
            </div>
          )}
        </div>
      )}

      {/* File List */}
      {files.length > 0 && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <h3 className="text-xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
            Uploaded Files
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {files.map((file, index) => (
              <div
                key={index}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '0.75rem 1rem',
                  backgroundColor: 'var(--surface-alt)',
                  borderRadius: '0.5rem',
                  gap: '1rem'
                }}
              >
                <div
                  style={{
                    width: '2.25rem',
                    height: '2.25rem',
                    backgroundColor: file.status === 'completed' ? '#ECFDF5' : file.status === 'error' ? '#FEF2F2' : 'var(--primary-100)',
                    borderRadius: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: file.status === 'completed' ? '#10B981' : file.status === 'error' ? '#EF4444' : 'var(--primary-600)'
                  }}
                >
                  {file.status === 'completed' ? <CheckCircle size={16} /> : 
                   file.status === 'error' ? <AlertCircle size={16} /> : <FileText size={16} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-primary)', fontWeight: '500', fontSize: '0.9rem' }}>{file.name}</span>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{file.size} MB</span>
                  </div>
                  {file.error && (
                    <span style={{ color: '#EF4444', fontSize: '0.75rem' }}>{file.error}</span>
                  )}
                  {file.status === 'uploading' && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                      <div style={{ flex: 1, height: '4px', backgroundColor: 'var(--surface)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', backgroundColor: 'var(--primary-500)', width: `${file.progress}%`, transition: 'width 0.3s ease' }} />
                      </div>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>{file.progress}%</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Controls */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h3 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
            Train Ensemble
          </h3>
          <button
            onClick={startTraining}
            disabled={isTraining}
            style={{
              backgroundColor: isTraining ? '#6B7280' : '#10B981',
              color: 'white',
              padding: '0.6rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              fontWeight: '500',
              cursor: isTraining ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            {isTraining ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={16} />}
            {isTraining ? 'Training...' : 'Start Training'}
          </button>
        </div>

        {trainingStatus && (
          <div style={{
            padding: '1rem',
            backgroundColor: 'var(--surface-alt)',
            borderRadius: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            {trainingStatus.status === 'trained' ? (
              <CheckCircle size={20} style={{ color: '#10B981' }} />
            ) : trainingStatus.status === 'error' ? (
              <AlertCircle size={20} style={{ color: '#EF4444' }} />
            ) : (
              <Loader2 size={20} style={{ color: 'var(--primary-500)', animation: 'spin 1s linear infinite' }} />
            )}
            <div>
              <div style={{ color: 'var(--text-primary)', fontWeight: '500', fontSize: '0.9rem' }}>
                {trainingStatus.message}
              </div>
              {trainingStatus.progress !== undefined && trainingStatus.progress < 100 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                  <div style={{ flex: 1, height: '6px', backgroundColor: 'var(--surface)', borderRadius: '3px', overflow: 'hidden', maxWidth: '300px' }}>
                    <div style={{ height: '100%', backgroundColor: 'var(--primary-500)', width: `${trainingStatus.progress}%`, transition: 'width 0.5s ease' }} />
                  </div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{trainingStatus.progress}%</span>
                </div>
              )}
            </div>
          </div>
        )}

        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '1rem' }}>
          Trains LSTM-GRU, Transformer, and XGBoost base models, then fits the Ridge stacker and GP residual model.
          Quick mode (~5 epochs) for testing. Upload or generate data first.
        </p>
      </div>
    </div>
  );
};

export default DataUpload;