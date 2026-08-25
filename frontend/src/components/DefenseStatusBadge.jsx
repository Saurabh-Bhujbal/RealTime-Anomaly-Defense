/**
 * DefenseStatusBadge
 * ──────────────────
 * Animated badge that communicates the defense verdict.
 *
 * Props:
 *   isAnomalous: boolean
 *   agreementScore: number  — 0–1 value from gradient_agreement
 */
import { ShieldCheck, ShieldAlert } from 'lucide-react'

export default function DefenseStatusBadge({ isAnomalous, agreementScore }) {
  const agreed = agreementScore >= 0.75

  if (isAnomalous) {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/15 border border-red-500/40 pulse-danger">
        <ShieldAlert size={16} className="text-red-400" />
        <span className="text-sm font-semibold text-red-400">Adversarial Input Detected</span>
      </div>
    )
  }

  if (!agreed) {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-yellow-500/15 border border-yellow-500/40">
        <ShieldAlert size={16} className="text-yellow-400" />
        <span className="text-sm font-semibold text-yellow-400">Low Defense Agreement</span>
      </div>
    )
  }

  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/15 border border-green-500/40">
      <ShieldCheck size={16} className="text-green-400" />
      <span className="text-sm font-semibold text-green-400">Defense Stable</span>
    </div>
  )
}
