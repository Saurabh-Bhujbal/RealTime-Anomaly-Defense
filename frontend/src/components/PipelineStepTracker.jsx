/**
 * PipelineStepTracker
 * ────────────────────
 * Visual stepper showing the 5-stage defense pipeline.
 * Each step highlights as 'active' or 'done' based on the
 * activeStep prop (0-indexed).
 *
 * Props:
 *   activeStep: number  — 0..4; -1 means nothing running
 */
import { Scan, Sparkles, Shuffle, GitFork, CheckCircle2 } from 'lucide-react'

const STEPS = [
  { icon: Scan,         label: 'Anomaly Detection' },
  { icon: Sparkles,     label: 'Self-Purification' },
  { icon: Shuffle,      label: 'Randomized Defense' },
  { icon: GitFork,      label: 'Gradient Diversity' },
  { icon: CheckCircle2, label: 'Final Prediction' },
]

export default function PipelineStepTracker({ activeStep = -1 }) {
  return (
    <div className="flex flex-col gap-2">
      {STEPS.map((step, idx) => {
        const done   = idx < activeStep
        const active = idx === activeStep

        return (
          <div
            key={idx}
            className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300
              ${active ? 'bg-brand-500/20 border border-brand-500/50' : ''}
              ${done   ? 'opacity-60' : ''}
              ${!active && !done ? 'opacity-30' : ''}
            `}
          >
            <step.icon
              size={18}
              className={
                active ? 'text-brand-500 animate-pulse' :
                done   ? 'text-green-400' :
                         'text-gray-600'
              }
            />
            <span className={`text-sm font-medium ${active ? 'text-gray-100' : 'text-gray-400'}`}>
              {step.label}
            </span>
            {done && (
              <CheckCircle2 size={14} className="ml-auto text-green-400" />
            )}
          </div>
        )
      })}
    </div>
  )
}
