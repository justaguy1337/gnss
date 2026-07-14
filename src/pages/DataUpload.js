import { useState, useRef, useEffect } from 'react';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  Play,
  Database,
  AlertCircle,
  Loader2,
  Server,
  ShieldCheck,
  Info
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const DataUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState(() => {
    try { return JSON.parse(localStorage.getItem('gnss_uploaded_files') || '[]'); } catch { return []; }
  });
  const [uploadResult, setUploadResult] = useState(() => {
    try { return JSON.parse(localStorage.getItem('gnss_upload_result') || 'null'); } catch { return null; }
  });
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const fileInputRef = useRef(null);

  // Poll server status on mount to show auto-train state
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`);
        if (res.ok) setServerStatus(await res.json());
      } catch { /* API not running yet */ }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 4000);
    return () => clearInterval(id);
  }, []);

  // Persist files and uploadResult to localStorage whenever they change
  useEffect(() => {
    // Only store serialisable fields (drop the raw File object)
    const serialisable = files.map(({ raw, ...rest }) => rest);
    localStorage.setItem('gnss_uploaded_files', JSON.stringify(serialisable));
  }, [files]);

  useEffect(() => {
    localStorage.setItem('gnss_upload_result', JSON.stringify(uploadResult));
  }, [uploadResult]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) return { success: true, data };
      return { success: false, error: data.detail || 'Upload failed' };
    } catch {
      return { success: false, error: 'API server not reachable. Start with: python ml/api.py' };
    }
  };

  const handleFiles = async (fileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      name: file.name,
      size: (file.size / 1024 / 1024).toFixed(2),
      status: 'uploading',
      progress: 0,
      raw: file,
      uploadedAt: new Date().toLocaleTimeString(),
    }));
    setFiles(prev => [...prev, ...newFiles]);

    for (const fileEntry of newFiles) {
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

      if (result.success) setUploadResult({ ...result.data, uploadedAt: new Date().toLocaleString() });
      else setUploadResult({ status: 'error', message: result.error });
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.length > 0) handleFiles(e.dataTransfer.files);
  };

  const handleFileSelect = (e) => {
    if (e.target.files?.length > 0) handleFiles(e.target.files);
  };

  const startTraining = async () => {
    setIsTraining(true);
    setTrainingStatus({ status: 'starting', message: 'Initiating retraining...' });
    try {
      const res = await fetch(`${API_BASE}/train?quick=true`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setTrainingStatus({ status: 'training', message: data.message });
        const poll = setInterval(async () => {
          try {
            const sr = await fetch(`${API_BASE}/status`);
            if (sr.ok) {
              const s = await sr.json();
              setTrainingStatus({ status: s.status, message: s.message, progress: s.progress });
              setServerStatus(s);
              if (s.status === 'trained' || s.status === 'error') {
                clearInterval(poll);
                setIsTraining(false);
              }
            }
          } catch { /* polling errors ignored */ }
        }, 3000);
      } else {
        const err = await res.json();
        setTrainingStatus({ status: 'error', message: err.detail });
        setIsTraining(false);
      }
    } catch {
      setTrainingStatus({ status: 'error', message: 'API server not reachable.' });
      setIsTraining(false);
    }
  };

  const schemaRows = [
    { col: 'utc_time',           type: 'string (datetime)', example: '9/8/2025 0:11',   note: 'M/D/YYYY H:MM format' },
    { col: 'x_error (m)',        type: 'float64',           example: '13.1117',          note: 'Ephemeris X error' },
    { col: 'y_error (m)',        type: 'float64',           example: '52.7899',          note: 'Ephemeris Y error' },
    { col: 'z_error (m)',        type: 'float64',           example: '-42.915',          note: 'Ephemeris Z error' },
    { col: 'satclockerror (m)',  type: 'float64',           example: '29.747',           note: 'Satellite clock error' },
  ];

  const statusColor = {
    trained: '#10B981', training: 'var(--primary-500)',
    error: '#EF4444',   initializing: '#F59E0B',
  };

  return (
    <div className="animate-fade-in">

      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Test Data Upload
        </h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Upload your GNSS test CSV for evaluation. Training data is loaded automatically
          from the ISRO dataset at server startup.
        </p>
      </div>

      {/* Server auto-train status banner */}
      <div
        className="card animate-slide-up"
        style={{
          marginBottom: '2rem',
          borderLeft: `4px solid ${statusColor[serverStatus?.status] || 'var(--border-color)'}`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <Server size={20} style={{ color: statusColor[serverStatus?.status] || 'var(--text-muted)' }} />
          <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
            Backend Status
          </span>
          {serverStatus && (
            <span style={{
              fontSize: '0.75rem',
              padding: '0.15rem 0.6rem',
              borderRadius: '999px',
              backgroundColor: statusColor[serverStatus.status] + '22',
              color: statusColor[serverStatus.status],
              fontWeight: '600',
              textTransform: 'uppercase',
            }}>
              {serverStatus.status}
            </span>
          )}
        </div>
        {serverStatus ? (
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginLeft: '2rem' }}>
            <p>{serverStatus.message}</p>
            {serverStatus.trained_satellites?.length > 0 && (
              <p style={{ marginTop: '0.25rem' }}>
                Trained satellites: <strong>{serverStatus.trained_satellites.join(', ')}</strong>
              </p>
            )}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginLeft: '2rem' }}>
            API server not reachable. Start it with: <code>python ml/api.py</code>
          </p>
        )}
      </div>

      <div className="grid-cols-2" style={{ marginBottom: '2rem' }}>

        {/* Upload Test CSV */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            Upload Test CSV
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
            Test data only — training data is auto-loaded at startup.
          </p>
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
              cursor: 'pointer',
            }}
          >
            <Upload size={32} style={{ color: 'var(--primary-500)', margin: '0 auto 0.75rem', display: 'block' }} />
            <h3 className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
              Drag & Drop Test CSV
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
              e.g. DATA_GEO_Test.csv or DATA_MEO_Test.csv
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          {/* Schema validation note */}
          <div style={{
            marginTop: '1rem',
            padding: '0.6rem 0.75rem',
            backgroundColor: 'var(--primary-50)',
            borderRadius: '0.5rem',
            display: 'flex',
            gap: '0.5rem',
            alignItems: 'flex-start',
          }}>
            <ShieldCheck size={15} style={{ color: 'var(--primary-600)', marginTop: '2px', flexShrink: 0 }} />
            <span style={{ color: 'var(--primary-700)', fontSize: '0.78rem' }}>
              Schema is validated automatically against training data before acceptance.
              A precise column diff is returned on mismatch.
            </span>
          </div>
        </div>

        {/* Training data info card */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            Training Data (Auto-loaded)
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
            Loaded from <code style={{ fontSize: '0.75rem' }}>dataset/</code> at server startup.
            Cannot be replaced via upload.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {[
              { file: 'DATA_GEO_Train.csv',  rows: '143 rows', period: 'Sep 1–7 2025', sat: 'GEO' },
              { file: 'DATA_MEO_Train.csv',  rows: '91 rows',  period: 'Sep 1–7 2025', sat: 'MEO (1)' },
              { file: 'DATA_MEO_Train2.csv', rows: '245 rows', period: 'Sep 3–9 2025', sat: 'MEO (2)' },
            ].map(f => (
              <div key={f.file} style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.5rem 0.75rem',
                backgroundColor: 'var(--surface-alt)',
                borderRadius: '0.5rem',
              }}>
                <Database size={15} style={{ color: '#10B981', flexShrink: 0 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ color: 'var(--text-primary)', fontWeight: '500', fontSize: '0.82rem' }}>
                    {f.file}
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                    {f.sat} · {f.rows} · {f.period}
                  </div>
                </div>
                <CheckCircle size={14} style={{ color: '#10B981' }} />
              </div>
            ))}
          </div>
          <div style={{
            marginTop: '1rem',
            padding: '0.6rem 0.75rem',
            backgroundColor: '#F0FDF4',
            borderRadius: '0.5rem',
            display: 'flex',
            gap: '0.5rem',
            alignItems: 'flex-start',
          }}>
            <Info size={14} style={{ color: '#16A34A', marginTop: '2px', flexShrink: 0 }} />
            <span style={{ color: '#15803D', fontSize: '0.78rem' }}>
              MEO train files are concatenated into a single MEO training series.
              GEO uses its own file. One GNSSEnsemble is trained per satellite type.
            </span>
          </div>
        </div>
      </div>

      {/* Upload result */}
      {uploadResult && (
        <div
          className="card"
          style={{
            marginBottom: '2rem',
            borderLeft: `4px solid ${uploadResult.status === 'success' ? '#10B981' : '#EF4444'}`,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            {uploadResult.status === 'success'
              ? <CheckCircle size={20} style={{ color: '#10B981' }} />
              : <AlertCircle size={20} style={{ color: '#EF4444' }} />}
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              {uploadResult.message}
            </span>
          </div>
          {uploadResult.summary && (
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginLeft: '2rem' }}>
              <p>{uploadResult.summary.n_satellites} satellite(s) — test rows updated</p>
              <p>{uploadResult.summary.n_meo} MEO + {uploadResult.summary.n_geo} GEO</p>
              {uploadResult.uploadedAt && (
                <p style={{ marginTop: '0.25rem', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  Uploaded at: {uploadResult.uploadedAt}
                </p>
              )}
              <p style={{
                marginTop: '0.5rem',
                padding: '0.35rem 0.6rem',
                backgroundColor: '#ECFDF5',
                borderRadius: '0.375rem',
                color: '#065F46',
                fontSize: '0.78rem',
                display: 'inline-block',
              }}>
                ✓ Dashboard predictions updated — switch to Dashboard tab to see results
              </p>
            </div>
          )}
        </div>
      )}

      {/* File list */}
      {files.length > 0 && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              Uploaded Files
            </h3>
            <button
              onClick={() => {
                setFiles([]);
                setUploadResult(null);
                localStorage.removeItem('gnss_uploaded_files');
                localStorage.removeItem('gnss_upload_result');
              }}
              style={{
                backgroundColor: 'transparent',
                border: '1px solid var(--border-color)',
                borderRadius: '0.375rem',
                padding: '0.3rem 0.75rem',
                color: 'var(--text-muted)',
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              Clear uploads
            </button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {files.map((file, idx) => (
              <div key={idx} style={{
                display: 'flex', alignItems: 'center', padding: '0.75rem 1rem',
                backgroundColor: 'var(--surface-alt)', borderRadius: '0.5rem', gap: '1rem',
              }}>
                <div style={{
                  width: '2.25rem', height: '2.25rem', borderRadius: '0.5rem',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  backgroundColor: file.status === 'completed' ? '#ECFDF5'
                    : file.status === 'error' ? '#FEF2F2' : 'var(--primary-100)',
                  color: file.status === 'completed' ? '#10B981'
                    : file.status === 'error' ? '#EF4444' : 'var(--primary-600)',
                }}>
                  {file.status === 'completed' ? <CheckCircle size={16} />
                    : file.status === 'error' ? <AlertCircle size={16} />
                    : <FileText size={16} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-primary)', fontWeight: '500', fontSize: '0.9rem' }}>
                      {file.name}
                    </span>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                      {file.size} MB
                    </span>
                  </div>
                  {file.error && (
                    <span style={{ color: '#EF4444', fontSize: '0.75rem', display: 'block', marginTop: '0.2rem' }}>
                      {file.error}
                    </span>
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

      {/* Retrain panel */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>
              Retrain Ensemble
            </h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.25rem' }}>
              Re-runs training on the ISRO dataset (runs in background, saves checkpoints)
            </p>
          </div>
          <button
            onClick={startTraining}
            disabled={isTraining || serverStatus?.status === 'initializing'}
            style={{
              backgroundColor: isTraining ? '#6B7280' : '#10B981',
              color: 'white',
              padding: '0.6rem 1.5rem',
              borderRadius: '0.5rem',
              border: 'none',
              fontWeight: '500',
              cursor: (isTraining || serverStatus?.status === 'initializing') ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              opacity: serverStatus?.status === 'initializing' ? 0.5 : 1,
            }}
          >
            {isTraining
              ? <><Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> Retraining...</>
              : <><Play size={16} /> Retrain</>}
          </button>
        </div>

        {trainingStatus && (
          <div style={{
            padding: '1rem', backgroundColor: 'var(--surface-alt)',
            borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem',
          }}>
            {trainingStatus.status === 'trained'
              ? <CheckCircle size={20} style={{ color: '#10B981' }} />
              : trainingStatus.status === 'error'
              ? <AlertCircle size={20} style={{ color: '#EF4444' }} />
              : <Loader2 size={20} style={{ color: 'var(--primary-500)', animation: 'spin 1s linear infinite' }} />}
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
      </div>

      {/* CSV schema reference */}
      <div className="card" style={{ marginTop: '2rem' }}>
        <h3 className="text-xl font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
          Expected CSV Schema
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1rem' }}>
          All dataset files (train and test) must match this schema. A mismatch returns a
          precise column diff error.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.83rem' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--surface-alt)' }}>
                {['Column Name', 'Type', 'Example Value', 'Description'].map(h => (
                  <th key={h} style={{
                    padding: '0.6rem 0.75rem', textAlign: 'left',
                    color: 'var(--text-secondary)', fontWeight: '600',
                    borderBottom: '1px solid var(--border-color)',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {schemaRows.map((row, i) => (
                <tr key={row.col} style={{ backgroundColor: i % 2 === 0 ? 'transparent' : 'var(--surface-alt)' }}>
                  <td style={{ padding: '0.6rem 0.75rem', fontFamily: 'monospace', color: 'var(--primary-600)', fontWeight: '500' }}>
                    {row.col}
                  </td>
                  <td style={{ padding: '0.6rem 0.75rem', color: 'var(--text-secondary)' }}>{row.type}</td>
                  <td style={{ padding: '0.6rem 0.75rem', fontFamily: 'monospace', color: 'var(--text-primary)', fontSize: '0.8rem' }}>
                    {row.example}
                  </td>
                  <td style={{ padding: '0.6rem 0.75rem', color: 'var(--text-muted)' }}>{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};

export default DataUpload;