import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [solution, setSolution] = useState<string | null>(null)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [noSolutionNeeded, setNoSolutionNeeded] = useState(false)
  const [rating, setRating] = useState<number>(5)
  const [ratingLoading, setRatingLoading] = useState(false)
  const [ratingMessage, setRatingMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)

    return () => {
      URL.revokeObjectURL(objectUrl)
    }
  }, [file])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      // reset state for new file
      setSolution(null)
      setThreadId(null)
      setNoSolutionNeeded(false)
      setRatingMessage(null)
      setError(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setSolution(null)
    setThreadId(null)
    setNoSolutionNeeded(false)
    setRatingMessage(null)

    const formData = new FormData()
    formData.append('traffic_frame', file)

    try {
      const response = await fetch('http://localhost:8000/image', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Failed to analyze frame: ${response.statusText}`)
      }

      const data = await response.json()
      
      if (data.message === "No solution needed") {
        setNoSolutionNeeded(true)
      } else {
        setSolution(data.message)
        setThreadId(data.thread_id)
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis.')
    } finally {
      setLoading(false)
    }
  }

  const handleRatingSubmit = async () => {
    if (!threadId) return

    setRatingLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rating, thread_id: threadId }),
      })

      if (!response.ok) {
        throw new Error(`Failed to submit rating: ${response.statusText}`)
      }

      const data = await response.json()
      setRatingMessage(data.message)
    } catch (err: any) {
      setError(err.message || 'An error occurred submitting feedback.')
    } finally {
      setRatingLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setSolution(null)
    setThreadId(null)
    setNoSolutionNeeded(false)
    setRatingMessage(null)
    setError(null)
    const fileInput = document.getElementById('file-upload') as HTMLInputElement
    if (fileInput) fileInput.value = ''
  }

  return (
    <div className="app-container">
      <header className="dashboard-header">
        <h1 className="dashboard-title">
          <span className="status-dot" />
          Traffic Analysis System
        </h1>
        <span className="system-badge">Vision AI Demo</span>
      </header>
      
      {error && (
        <div className="alert-error">
          <strong>Error: </strong>{error}
        </div>
      )}

      {/* Frame Preview - Persists throughout analysis/solution/rating flow */}
      {previewUrl && (
        <div className="preview-wrapper">
          <div className="preview-header">
            <span>Current Frame: {file?.name}</span>
            <span>{(file ? (file.size / 1024).toFixed(1) + ' KB' : '')}</span>
          </div>
          <div className="preview-image-container">
            <img src={previewUrl} alt="Traffic Frame Preview" className="preview-image" />
          </div>
        </div>
      )}

      {!solution && !noSolutionNeeded && (
        <div className="panel">
          <div className="form-group">
            <label htmlFor="file-upload" className="form-label">
              Upload Traffic Frame (JPG / PNG / WEBP)
            </label>
            <input 
              id="file-upload"
              type="file" 
              accept=".jpg,.jpeg,.png,.webp" 
              onChange={handleFileChange}
              disabled={loading}
              className="file-input"
            />
          </div>
          
          <button 
            onClick={handleAnalyze} 
            disabled={!file || loading}
            className="btn btn-primary"
          >
            {loading && <span className="spinner" />}
            {loading ? 'Analyzing Frame...' : 'Analyze Frame'}
          </button>
        </div>
      )}

      {noSolutionNeeded && (
        <div className="status-panel-no-solution">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <h2>No Traffic Issue Detected</h2>
            <span className="status-badge-success">Clear</span>
          </div>
          <p style={{ margin: '0 0 1.25rem 0', color: 'var(--text-secondary)' }}>
            No solution needed for this traffic frame.
          </p>
          <button 
            onClick={handleReset} 
            className="btn btn-outline"
          >
            Upload New Frame
          </button>
        </div>
      )}

      {solution && (
        <div className="solution-panel">
          <div className="solution-header">
            <h2>Proposed Solution</h2>
            <span className="status-badge-warning">Action Required</span>
          </div>
          
          <div className="solution-content">
            {solution}
          </div>

          {!ratingMessage ? (
            <div className="rating-section">
              <h3>Rate this solution</h3>
              <div className="rating-controls">
                <select 
                  value={rating} 
                  onChange={(e) => setRating(Number(e.target.value))}
                  disabled={ratingLoading}
                  className="select-input"
                >
                  <option value={5}>5 - Excellent</option>
                  <option value={4}>4 - Good</option>
                  <option value={3}>3 - Okay</option>
                  <option value={2}>2 - Poor</option>
                  <option value={1}>1 - Terrible</option>
                </select>
                
                <button 
                  onClick={handleRatingSubmit} 
                  disabled={ratingLoading}
                  className="btn btn-success"
                >
                  {ratingLoading && <span className="spinner" />}
                  {ratingLoading ? 'Submitting...' : 'Submit Rating'}
                </button>
              </div>
            </div>
          ) : (
            <div className="feedback-success">
              <div>
                <strong>Feedback Response: </strong>
                <span>{ratingMessage}</span>
              </div>
              <div style={{ marginTop: '1.25rem' }}>
                <button 
                  onClick={handleReset} 
                  className="btn btn-outline"
                >
                  Upload New Frame
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
