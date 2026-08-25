/**
 * AnalyticsPage  (Option B — per-user personal analytics)
 * ─────────────────────────────────────────────────────────
 * Fetches the authenticated user's own inference history from
 * GET /api/history and builds charts entirely from their data:
 *
 *   • Summary stat cards (total, anomaly rate, avg confidence, avg exec time)
 *   • Prediction distribution bar chart (digit 0–9 frequency)
 *   • Anomaly score trend line chart (chronological)
 *   • Defense agreement histogram
 *   • Clean vs Attacked pie chart
 *   • Recent logs table
 *
 * Empty / unauthenticated states are handled gracefully.
 */
import { useEffect, useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, LineChart, Line,
  PieChart, Pie, Cell,
} from 'recharts'
import {
  Activity, ShieldCheck, BarChart2,
  AlertTriangle, Database, Clock, TrendingUp,
} from 'lucide-react'

import client from '../api/client'
import MetricCard from '../components/MetricCard'
import { useAuth } from '../context/AuthContext'

// Tiny native date formatter — avoids adding date-fns dependency
const fmtDate = (iso) =>
  new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
  })
const fmtTime = (iso) =>
  new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })

// ── Colour tokens ────────────────────────────────────────────────────────────
const C_CLEAN    = '#22c55e'
const C_ATTACKED = '#ef4444'
const C_ANOMALY  = '#f59e0b'
const C_DEFENSE  = '#6366f1'
const C_DIGIT    = '#38bdf8'

const DIGIT_COLORS = [
  '#6366f1','#8b5cf6','#a78bfa','#38bdf8','#22c55e',
  '#f59e0b','#ef4444','#ec4899','#14b8a6','#fb923c',
]

// ── Shared chart style helpers ────────────────────────────────────────────────
const axisStyle   = { fill: '#9ca3af', fontSize: 11 }
const gridStyle   = { strokeDasharray: '3 3', stroke: '#1f2937' }
const tooltipCss  = {
  contentStyle : { background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 12 },
  labelStyle   : { color: '#e5e7eb' },
  itemStyle    : { color: '#d1d5db' },
}

// ── Wrappers ──────────────────────────────────────────────────────────────────
function Section({ title, icon: Icon, children, className = '' }) {
  return (
    <div className={`glass p-6 rounded-xl flex flex-col gap-4 ${className}`}>
      <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <Icon size={16} className="text-brand-400" />
        {title}
      </h2>
      {children}
    </div>
  )
}

// ── Custom tooltip shared ─────────────────────────────────────────────────────
const CustomTip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      {label && <p className="text-gray-400 font-medium mb-1">{label}</p>}
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color ?? p.fill }}>
          {p.name}: <span className="font-bold">{p.value}</span>
        </p>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AnalyticsPage() {
  const { user } = useAuth()
  const [items,   setItems]   = useState([])
  const [total,   setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(false)

  // Fetch ALL the user's history (up to 200 rows for charting)
  useEffect(() => {
    if (!user) { setLoading(false); return }

    client.get('/api/history', { params: { page: 1, page_size: 200 } })
      .then((res) => {
        setItems(res.data.items ?? [])
        setTotal(res.data.total ?? 0)
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false))
  }, [user])

  // ── Derived data (memoised) ──────────────────────────────────────────────
  const stats = useMemo(() => {
    if (!items.length) return null

    const anomalousCount = items.filter((i) => i.is_anomalous).length
    const attackedCount  = items.filter((i) => i.is_fgsm_attacked).length
    const avgConf        = items.reduce((s, i) => s + i.clean_confidence, 0) / items.length
    const avgExec        = items.reduce((s, i) => s + i.execution_time_ms, 0) / items.length
    const avgAgreement   = items.reduce((s, i) => s + i.defense_agreement_score, 0) / items.length

    return { anomalousCount, attackedCount, avgConf, avgExec, avgAgreement }
  }, [items])

  // Digit 0–9 frequency (of final_defended_prediction)
  const digitData = useMemo(() => {
    const counts = Array.from({ length: 10 }, (_, d) => ({ digit: `${d}`, count: 0 }))
    items.forEach((i) => { counts[i.final_defended_prediction].count++ })
    return counts
  }, [items])

  // Anomaly score over time (last 50)
  const anomalyTrend = useMemo(() =>
    [...items]
      .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
      .slice(-50)
      .map((i, idx) => ({
        idx: idx + 1,
        score:     parseFloat(i.anomaly_score.toFixed(4)),
        threshold: 0.0379,           // calibrated threshold from detector
        label:     fmtTime(i.created_at),
      })),
    [items]
  )

  // Defense agreement histogram (buckets: 0-25%, 25-50%, 50-75%, 75-100%)
  const agreementHist = useMemo(() => {
    const buckets = [
      { range: '0–25%',  count: 0 },
      { range: '25–50%', count: 0 },
      { range: '50–75%', count: 0 },
      { range: '75–100%',count: 0 },
    ]
    items.forEach((i) => {
      const pct = i.defense_agreement_score * 100
      if      (pct < 25)  buckets[0].count++
      else if (pct < 50)  buckets[1].count++
      else if (pct < 75)  buckets[2].count++
      else                buckets[3].count++
    })
    return buckets
  }, [items])

  // Clean vs Attacked pie
  const pieData = useMemo(() => {
    if (!stats) return []
    return [
      { name: 'Clean',   value: items.length - stats.attackedCount, fill: C_CLEAN   },
      { name: 'Attacked',value: stats.attackedCount,                fill: C_ATTACKED },
    ]
  }, [items, stats])

  // Recent 10 rows
  const recentRows = useMemo(() =>
    [...items]
      .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      .slice(0, 10),
    [items]
  )

  // ── States ───────────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-gray-500">
        <ShieldCheck size={48} className="text-brand-500 opacity-50" />
        <p className="text-sm">Sign in to view your personal analytics.</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-gray-500">
        <span className="animate-spin rounded-full h-10 w-10 border-2 border-brand-500/30 border-t-brand-500" />
        <p className="text-sm">Loading your analytics…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 text-red-400">
        <AlertTriangle size={40} />
        <p className="text-sm">Failed to load history. Is the backend running?</p>
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-16 flex flex-col items-center gap-4 text-gray-400">
        <Database size={48} className="text-brand-500 opacity-60" />
        <h2 className="text-xl font-semibold text-gray-200">No Analyses Yet</h2>
        <p className="text-sm text-gray-500 text-center max-w-md">
          Your personal analytics will appear here after you upload and analyze
          your first image on the <span className="text-brand-400">Live Detection</span> page.
        </p>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col gap-8">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">My Analytics</h1>
        <p className="text-gray-500 mt-1 text-sm">
          Personal insights from your {total} image{total !== 1 ? 's' : ''} analyzed
          {total > 200 && ' (showing most recent 200)'}
        </p>
      </div>

      {/* ── Summary stat cards ────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <MetricCard
          label="Total Analyses"
          value={total}
          highlight="default"
        />
        <MetricCard
          label="Anomaly Rate"
          value={`${((stats.anomalousCount / items.length) * 100).toFixed(1)}%`}
          highlight={stats.anomalousCount / items.length > 0.3 ? 'danger' : 'success'}
        />
        <MetricCard
          label="Avg Confidence"
          value={`${(stats.avgConf * 100).toFixed(1)}%`}
          highlight={stats.avgConf > 0.85 ? 'success' : 'warning'}
        />
        <MetricCard
          label="Avg Agreement"
          value={`${(stats.avgAgreement * 100).toFixed(0)}%`}
          highlight={stats.avgAgreement >= 0.75 ? 'success' : 'warning'}
        />
        <MetricCard
          label="Avg Exec Time"
          value={`${stats.avgExec.toFixed(0)}ms`}
          highlight="default"
        />
      </div>

      {/* ── Row: Prediction Distribution + Clean vs Attacked ──────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        <Section title="Prediction Distribution (Digit 0–9)" icon={BarChart2} className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={digitData} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
              <CartesianGrid {...gridStyle} />
              <XAxis dataKey="digit" tick={axisStyle} />
              <YAxis tick={axisStyle} allowDecimals={false} />
              <Tooltip {...tooltipCss} content={<CustomTip />} />
              <Bar dataKey="count" name="Predictions" radius={[4, 4, 0, 0]}>
                {digitData.map((_, i) => (
                  <Cell key={i} fill={DIGIT_COLORS[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-600 text-center">
            Frequency of each final defended prediction across all your uploads.
          </p>
        </Section>

        <Section title="Clean vs Attacked" icon={ShieldCheck}>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={4}
                dataKey="value"
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={tooltipCss.contentStyle}
                itemStyle={tooltipCss.itemStyle}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-around text-xs text-gray-400">
            <span>
              <span className="text-green-400 font-bold">{items.length - stats.attackedCount}</span> clean
            </span>
            <span>
              <span className="text-red-400 font-bold">{stats.attackedCount}</span> attacked
            </span>
          </div>
        </Section>
      </div>

      {/* ── Anomaly Score Trend ───────────────────────────────────────────── */}
      {anomalyTrend.length > 1 && (
        <Section title="Anomaly Score Over Time (last 50 analyses)" icon={Activity}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={anomalyTrend} margin={{ top: 4, right: 16, left: -10, bottom: 4 }}>
              <CartesianGrid {...gridStyle} />
              <XAxis dataKey="label" tick={{ ...axisStyle, fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis tick={axisStyle} domain={[0, 'auto']} />
              <Tooltip {...tooltipCss} content={<CustomTip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
              <Line
                type="monotone"
                dataKey="score"
                name="Anomaly Score"
                stroke={C_ANOMALY}
                strokeWidth={2}
                dot={{ r: 3, fill: C_ANOMALY }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="threshold"
                name="Detector Threshold"
                stroke="#6b7280"
                strokeWidth={1}
                strokeDasharray="6 3"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-600 text-center">
            Scores above the dashed threshold are flagged as adversarial anomalies.
          </p>
        </Section>
      )}

      {/* ── Defense Agreement Histogram ───────────────────────────────────── */}
      <Section title="Defense Agreement Distribution" icon={TrendingUp}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={agreementHist} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
            <CartesianGrid {...gridStyle} />
            <XAxis dataKey="range" tick={axisStyle} />
            <YAxis tick={axisStyle} allowDecimals={false} />
            <Tooltip {...tooltipCss} content={<CustomTip />} />
            <Bar dataKey="count" name="# Analyses" fill={C_DEFENSE} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-600 text-center">
          Higher agreement (75–100%) means all defense heads agree — a reliable prediction.
        </p>
      </Section>

      {/* ── Recent logs table ─────────────────────────────────────────────── */}
      <Section title="Recent Analyses" icon={Clock}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase tracking-wider">
                <th className="py-2 pr-4">Time</th>
                <th className="py-2 pr-4">Prediction</th>
                <th className="py-2 pr-4">Confidence</th>
                <th className="py-2 pr-4">Anomaly Score</th>
                <th className="py-2 pr-4">Attacked?</th>
                <th className="py-2 pr-4">Agreement</th>
                <th className="py-2">Exec</th>
              </tr>
            </thead>
            <tbody>
              {recentRows.map((r) => (
                <tr key={r.id} className="border-b border-gray-800/50 hover:bg-white/5 transition-colors">
                  <td className="py-2 pr-4 text-gray-500 font-mono whitespace-nowrap">
                    {fmtDate(r.created_at)}
                  </td>
                  <td className="py-2 pr-4">
                    <span className="text-brand-400 font-bold text-sm">{r.final_defended_prediction}</span>
                    <span className="text-gray-600 ml-1">(raw: {r.clean_prediction})</span>
                  </td>
                  <td className="py-2 pr-4 text-gray-300">
                    {(r.clean_confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 pr-4">
                    <span className={r.is_anomalous ? 'text-red-400 font-semibold' : 'text-gray-400'}>
                      {r.anomaly_score.toFixed(4)}
                      {r.is_anomalous && ' ⚠️'}
                    </span>
                  </td>
                  <td className="py-2 pr-4">
                    {r.is_fgsm_attacked ? (
                      <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-red-900/40 text-red-400">
                        {r.attack_type?.toUpperCase()} ε={r.attack_epsilon?.toFixed(2)}
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-green-900/30 text-green-400">
                        Clean
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4">
                    <span className={
                      r.defense_agreement_score >= 0.75 ? 'text-green-400' : 'text-amber-400'
                    }>
                      {(r.defense_agreement_score * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-2 text-gray-500 font-mono">
                    {r.execution_time_ms.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {total > 10 && (
            <p className="text-xs text-gray-600 mt-3 text-center">
              Showing last 10 of {total} total analyses.
            </p>
          )}
        </div>
      </Section>

    </div>
  )
}
