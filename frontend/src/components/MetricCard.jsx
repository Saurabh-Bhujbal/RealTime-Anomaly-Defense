/**
 * MetricCard
 * ──────────
 * Glassmorphism card displaying a single numeric metric.
 *
 * Props:
 *   label:    string  — e.g. "Clean Accuracy"
 *   value:    string  — e.g. "98.94%" or "7"
 *   icon?:    ReactNode
 *   sub?:     string  — small subtitle below value
 *   highlight?: 'default' | 'success' | 'danger' | 'warning'
 */
export default function MetricCard({ label, value, icon, sub, highlight = 'default' }) {
  const ringColor = {
    default: 'border-gray-700/60',
    success: 'border-green-500/50',
    danger:  'border-red-500/50',
    warning: 'border-yellow-500/50',
  }[highlight]

  const valueColor = {
    default: 'text-gray-100',
    success: 'text-green-400',
    danger:  'text-red-400',
    warning: 'text-yellow-400',
  }[highlight]

  return (
    <div className={`glass p-5 flex flex-col gap-1 border ${ringColor} transition-all duration-200 hover:border-brand-500/40`}>
      <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-widest font-medium">
        {icon && <span className="text-brand-500">{icon}</span>}
        {label}
      </div>
      <div className={`text-3xl font-bold font-mono ${valueColor}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-1">{sub}</div>}
    </div>
  )
}
