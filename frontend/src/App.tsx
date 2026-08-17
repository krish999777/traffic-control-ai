import { useState } from 'react'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [solution, setSolution] = useState<string | null>(null)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [noSolutionNeeded, setNoSolutionNeeded] = useState(false)
  const [rating, setRating] = useState<number>(5)
  const [ratingLoading, setRatingLoading] = useState(false)
  const [ratingMessage, setRatingMessage] = useState<string | null>(null)

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
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Traffic Analysis Demo</h1>
      
      {error && (
        <div style={{ color: '#d32f2f', backgroundColor: '#ffebee', padding: '1rem', border: '1px solid #ef9a9a', borderRadius: '4px', marginBottom: '1rem' }}>
          <strong>Error: </strong>{error}
        </div>
      )}

      {!solution && !noSolutionNeeded && (
        <div style={{ marginBottom: '2rem', padding: '1.5rem', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
          <div style={{ marginBottom: '1rem' }}>
            <label htmlFor="file-upload" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Upload Traffic Frame (JPG/PNG/WEBP):
            </label>
            <input 
              id="file-upload"
              type="file" 
              accept=".jpg,.jpeg,.png,.webp" 
              onChange={handleFileChange}
              disabled={loading}
              style={{ display: 'block', width: '100%', padding: '0.5rem' }}
            />
          </div>
          
          <button 
            onClick={handleAnalyze} 
            disabled={!file || loading}
            style={{ 
              padding: '0.75rem 1.5rem', 
              cursor: (!file || loading) ? 'not-allowed' : 'pointer',
              backgroundColor: (!file || loading) ? '#cccccc' : '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              fontSize: '1rem',
              fontWeight: 'bold'
            }}
          >
            {loading ? 'Analyzing Frame...' : 'Analyze Frame'}
          </button>
        </div>
      )}

      {noSolutionNeeded && (
        <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#e8f5e9', borderRadius: '8px', border: '1px solid #a5d6a7' }}>
          <h2 style={{ color: '#2e7d32', marginTop: 0 }}>No traffic issue detected. No solution needed.</h2>
          <button 
            onClick={handleReset} 
            style={{ 
              padding: '0.5rem 1rem', 
              marginTop: '1rem',
              backgroundColor: '#ffffff',
              border: '1px solid #ccc',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Upload New Frame
          </button>
        </div>
      )}

      {solution && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Proposed Solution</h2>
          <div style={{ backgroundColor: '#fff3e0', padding: '1.5rem', borderRadius: '8px', border: '1px solid #ffcc80', whiteSpace: 'pre-wrap', marginBottom: '2rem' }}>
            {solution}
          </div>

          {!ratingMessage ? (
            <div style={{ padding: '1.5rem', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
              <h3 style={{ marginTop: 0 }}>Rate this solution</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <select 
                  value={rating} 
                  onChange={(e) => setRating(Number(e.target.value))}
                  disabled={ratingLoading}
                  style={{ padding: '0.5rem', fontSize: '1rem', borderRadius: '4px', border: '1px solid #ccc' }}
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
                  style={{ 
                    padding: '0.5rem 1rem',
                    backgroundColor: ratingLoading ? '#cccccc' : '#4caf50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: ratingLoading ? 'not-allowed' : 'pointer',
                    fontSize: '1rem'
                  }}
                >
                  {ratingLoading ? 'Submitting...' : 'Submit Rating'}
                </button>
              </div>
            </div>
          ) : (
            <div style={{ padding: '1.5rem', backgroundColor: '#e8f5e9', borderRadius: '8px', border: '1px solid #a5d6a7' }}>
              <strong style={{ color: '#2e7d32' }}>Feedback Response: </strong> 
              <span>{ratingMessage}</span>
              <div style={{ marginTop: '1.5rem' }}>
                <button 
                  onClick={handleReset} 
                  style={{ 
                    padding: '0.5rem 1rem',
                    backgroundColor: '#ffffff',
                    border: '1px solid #ccc',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
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
