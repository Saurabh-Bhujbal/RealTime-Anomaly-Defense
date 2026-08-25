/**
 * HistoryPage
 * ───────────
 * Shows the authenticated user's past inference logs.
 * Paginated via page / page_size query params.
 */
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ShieldCheck, ShieldAlert, ChevronLeft, ChevronRight, History } from 'lucide-react'
import client from '../api/client'
import { useAuth } from '../context/AuthContext'

const PAGE_SIZE = 15

export default function HistoryPage() {
  const { user } = useAuth()
  const navigate = useNavigate()

  const [items, setItems]   = useState([])
  const [total, setTotal]   = useState(0)
  const [page, setPage]     = useState(1)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!user) { navigate('/login'); return }
    setLoading(true)
    client.get(`/api/history?page=${page}&page_size=${PAGE_SIZE}`)
      .then((res) => { setItems(res.data.items); setTotal(res.data.total) })
      .catch(() => setItems([]))
      .finally(() => setLoading(false))
  }, [user, page])

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const fmt = (iso) => new Date(iso).toLocaleString()

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Inspection History</h1>
          <p className="text-gray-500 mt-1 text-sm">Your past image analyses ({total} total)</p>
        </div>
        <History size={28} className="text-brand-400" />
      </div>

      {loading ? (
        <div className="flex justify-center py-24 text-gray-600">Loading…</div>
      ) : items.length === 0 ? (
        <div className="glass p-12 text-center text-gray-600">
          No analyses yet. Upload an image on the Dashboard to get started.
        </div>
      ) : (
        <>
          <div className="glass overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-xs text-gray-600 uppercase tracking-wider">
                  {['Date','Clean Pred.','Confidence','Anomaly?','Attack','Defended','Agreement','Time (ms)'].map((h) => (
                    <th key={h} className="text-left px-4 py-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-b border-gray-800/60 hover:bg-white/[0.02] transition-colors">
                    <td className="px-4 py-3 text-gray-500 whitespace-nowrap font-mono text-xs">
                      {fmt(item.created_at)}
                    </td>
                    <td className="px-4 py-3 font-mono font-bold text-gray-200">{item.clean_prediction}</td>
                    <td className="px-4 py-3 font-mono text-gray-400">
                      {(item.clean_confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3">
                      {item.is_anomalous
                        ? <ShieldAlert size={16} className="text-red-400" />
                        : <ShieldCheck size={16} className="text-green-400" />}
                    </td>
                    <td className="px-4 py-3">
                      {item.is_fgsm_attacked
                        ? <span className="px-2 py-0.5 rounded bg-red-500/15 text-red-400 text-xs font-mono uppercase">
                            {item.attack_type} ε={item.attack_epsilon}
                          </span>
                        : <span className="text-gray-700">—</span>}
                    </td>
                    <td className="px-4 py-3 font-mono font-bold text-green-400">{item.final_defended_prediction}</td>
                    <td className="px-4 py-3 font-mono text-gray-400">
                      {(item.defense_agreement_score * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-3 font-mono text-gray-600">
                      {item.execution_time_ms.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-30 transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm text-gray-500">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-30 transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </>
      )}
    </div>
  )
}
