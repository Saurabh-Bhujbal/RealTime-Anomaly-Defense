/**
 * EpsilonSlider
 * ─────────────
 * Range slider for selecting FGSM / PGD epsilon (perturbation strength).
 *
 * Props:
 *   value:    number
 *   onChange: (value: number) => void
 *   disabled?: boolean
 */
const STEPS = [0.05, 0.10, 0.20]

export default function EpsilonSlider({ value, onChange, disabled = false }) {
  const idx = STEPS.indexOf(value)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-400 font-medium">
          Attack Strength (ε)
        </label>
        <span className="text-sm font-mono font-bold text-brand-400">
          ε = {value.toFixed(2)}
        </span>
      </div>

      {/* Custom three-stop slider */}
      <div className="relative flex items-center gap-3">
        {STEPS.map((step) => (
          <button
            key={step}
            onClick={() => !disabled && onChange(step)}
            disabled={disabled}
            className={`flex-1 py-2 rounded-lg text-xs font-mono font-semibold transition-all duration-200
              ${value === step
                ? 'bg-brand-600 text-white shadow-lg shadow-brand-500/30'
                : 'bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-300'}
              ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}
            `}
          >
            {step.toFixed(2)}
          </button>
        ))}
      </div>

      <div className="flex justify-between text-xs text-gray-600">
        <span>Weak</span>
        <span>Strong</span>
      </div>
    </div>
  )
}
