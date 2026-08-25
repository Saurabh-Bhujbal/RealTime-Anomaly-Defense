/**
 * DashboardPage
 * ─────────────
 * The main live detection page.
 *
 * Flow:
 *   1. User uploads an image (ImageUploader)
 *   2. Clicks "Analyze" → POST /api/defense/analyze (requires auth)
 *   3. Results appear as MetricCards + DefenseStatusBadge
 *   4. Optionally: choose attack type + ε, click "Run Attack Test"
 *      → POST /api/defense/attack-test (requires auth)
 *   5. Adversarial results appear alongside clean results
 *
 * Access model: this page loads for everyone (it is the app's landing page),
 * but every action that hits the ML pipeline requires login. Uploading an
 * image, clicking "Analyze", or running an attack test while signed out
 * redirects the visitor to /login. The backend independently enforces auth
 * on /api/defense/analyze and /api/defense/attack-test.
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import { Zap, Swords } from 'lucide-react'

import client from '../api/client'
import { useAuth } from '../context/AuthContext'

import ImageUploader       from '../components/ImageUploader'
import MetricCard          from '../components/MetricCard'
import DefenseStatusBadge  from '../components/DefenseStatusBadge'
import PipelineStepTracker from '../components/PipelineStepTracker'
import EpsilonSlider       from '../components/EpsilonSlider'

export default function DashboardPage() {
  const { user } = useAuth()
  const navigate = useNavigate()

  const [file, setFile]               = useState(null)
  const [pipelineStep, setPipelineStep] = useState(-1)
  const [result, setResult]           = useState(null)
  const [attackResult, setAttackResult] = useState(null)
  const [loading, setLoading]         = useState(false)
  const [attackLoading, setAttackLoading] = useState(false)
  const [attackType, setAttackType]   = useState('fgsm')
  const [epsilon, setEpsilon]         = useState(0.20)

  // ── Login gate ───────────────────────────────────────────────────────────────
  // Analysis actions require a signed-in user; if not, bounce to /login.
  const requireLogin = () => {
    if (!user) {
      toast.error('Please sign in to analyze images.')
      navigate('/login')
      return false
    }
    return true
  }

  // Gate the drag-and-drop path: block file selection when signed out.
  const handleFileSelect = (selected) => {
    if (selected && !requireLogin()) return
    setFile(selected)
  }

  // ── Analyze clean image ──────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!requireLogin()) return
    if (!file) { toast.error('Please upload an image first.'); return }
    setLoading(true)
    setResult(null)
    setAttackResult(null)
    setPipelineStep(1)

    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await client.post('/api/defense/analyze', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
      setPipelineStep(5)
      toast.success('Analysis complete!')
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Analysis failed.')
      setPipelineStep(-1)
    } finally {
      setLoading(false)
    }
  }

  // ── Attack test ──────────────────────────────────────────────────────────────
  const handleAttackTest = async () => {
    if (!requireLogin()) return
    if (!file)  { toast.error('Upload an image first.'); return }
    setAttackLoading(true)
    setAttackResult(null)

    try {
      const form = new FormData()
      form.append('file', file)
      form.append('attack_type', attackType)
      form.append('epsilon', epsilon)
      const { data } = await client.post('/api/defense/attack-test', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setAttackResult(data)
      toast.success('Attack test complete!')
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Attack test failed.')
    } finally {
      setAttackLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col gap-8">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Live Detection</h1>
        <p className="text-gray-500 mt-1 text-sm">
          Upload a handwritten digit image and run the full adversarial defense pipeline.
        </p>
      </div>

      {/* ── Upload + Pipeline ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 flex flex-col gap-4">
          {/* Signed-out users are bounced to /login on interaction. onClickCapture
              handles the click-to-browse path; drag-drop is gated in handleFileSelect. */}
          <div onClickCapture={(e) => {
            if (!user) { e.preventDefault(); e.stopPropagation(); requireLogin() }
          }}>
            <ImageUploader onFileSelect={handleFileSelect} disabled={loading} />
          </div>
          <button
            id="analyze-btn"
            onClick={handleAnalyze}
            disabled={user ? (loading || !file) : false}
            className="flex items-center justify-center gap-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white font-semibold py-3 rounded-xl transition-all duration-200 shadow-lg shadow-brand-500/20"
          >
            {loading ? (
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white" />
            ) : (
              <Zap size={16} />
            )}
            {!user ? 'Sign in to Analyze' : (loading ? 'Running Pipeline…' : 'Analyze Image')}
          </button>
        </div>

        <div className="glass p-5">
          <h3 className="text-xs text-gray-500 uppercase tracking-widest font-medium mb-4">
            Defense Pipeline
          </h3>
          <PipelineStepTracker activeStep={loading ? pipelineStep : (result ? 5 : -1)} />
        </div>
      </div>

      {/* ── Clean Results ─────────────────────────────────────────────────── */}
      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-200">Analysis Results</h2>
            <DefenseStatusBadge
              isAnomalous={result.is_anomalous}
              agreementScore={result.defense_agreement_score}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            <MetricCard label="Predicted Digit"     value={result.clean_prediction}                              />
            <MetricCard label="Confidence"          value={`${(result.clean_confidence * 100).toFixed(1)}%`}     highlight={result.clean_confidence > 0.9 ? 'success' : 'warning'} />
            <MetricCard label="Anomaly Score"       value={result.anomaly_score.toFixed(4)}                      highlight={result.is_anomalous ? 'danger' : 'success'} />
            <MetricCard label="Final Prediction"    value={result.final_prediction}                               highlight="success" />
            <MetricCard label="Randomized Pred."   value={result.randomized_prediction}                          />
            <MetricCard label="Gradient Pred."     value={result.gradient_prediction}                            />
            <MetricCard label="Agreement"          value={`${(result.defense_agreement_score * 100).toFixed(0)}%`} highlight={result.defense_agreement_score >= 0.75 ? 'success' : 'warning'} />
            <MetricCard label="Execution Time"     value={`${result.execution_time_ms.toFixed(0)}ms`}            />
          </div>
        </div>
      )}

      {/* ── Adversarial Test Section ──────────────────────────────────────── */}
      <div className="glass p-6 flex flex-col gap-5">
        <div>
          <h2 className="text-lg font-semibold text-gray-200">⚔️ Adversarial Attack Test</h2>
          <p className="text-sm text-gray-500 mt-1">
            Generate a real adversarial example from the uploaded image and test the defense pipeline.
            {!user && <span className="text-yellow-400"> (Sign in required)</span>}
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {/* Attack type */}
          <div className="flex flex-col gap-2">
            <label className="text-xs text-gray-400 uppercase tracking-wider font-medium">Attack Type</label>
            <div className="flex gap-2">
              {['fgsm','pgd','eot_pgd'].map((t) => (
                <button
                  key={t}
                  onClick={() => setAttackType(t)}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-semibold transition-all
                    ${attackType === t ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-500 hover:bg-gray-700'}`}
                >
                  {t.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <EpsilonSlider value={epsilon} onChange={setEpsilon} disabled={attackLoading} />
        </div>

        <button
          id="attack-test-btn"
          onClick={handleAttackTest}
          disabled={user ? (attackLoading || !file) : false}
          className="flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-semibold py-3 rounded-xl transition-all duration-200"
        >
          {attackLoading ? (
            <span className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white" />
          ) : (
            <Swords size={16} />
          )}
          {!user ? 'Sign in to Run Attack' : (attackLoading ? 'Generating Attack…' : `Run ${attackType.toUpperCase()} Attack`)}
        </button>

        {attackResult && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mt-2">
            <MetricCard label="Clean Digit"          value={attackResult.clean_prediction}                                        />
            <MetricCard label="Adversarial Digit"    value={attackResult.adversarial_prediction}   highlight={attackResult.adversarial_prediction !== attackResult.clean_prediction ? 'danger' : 'success'} />
            <MetricCard label="Defended Prediction"  value={attackResult.final_defended_prediction} highlight="success"            />
            <MetricCard label="Defense Agreement"    value={`${(attackResult.defense_agreement_score * 100).toFixed(0)}%`}         highlight={attackResult.defense_agreement_score >= 0.75 ? 'success' : 'danger'} />
            <MetricCard label="Anomaly Score"        value={attackResult.anomaly_score.toFixed(4)}  highlight={attackResult.is_anomalous ? 'danger' : 'success'} />
            <MetricCard label="Attack ε"             value={attackResult.epsilon.toFixed(2)}                                       />
            <MetricCard label="Exec. Time"           value={`${attackResult.execution_time_ms.toFixed(0)}ms`}                      />
          </div>
        )}
      </div>
    </div>
  )
}
