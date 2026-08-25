import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { Shield, Mail, Lock, UserPlus } from 'lucide-react'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate      = useNavigate()
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await register(email, password)
      toast.success('Account created! Please sign in.')
      navigate('/login')
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Registration failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-96 h-96 bg-brand-600/20 rounded-full blur-3xl" />
      </div>

      <div className="glass w-full max-w-md p-8 z-10">
        <div className="flex flex-col items-center gap-2 mb-8">
          <div className="p-3 bg-brand-600/20 rounded-xl">
            <Shield size={28} className="text-brand-400" />
          </div>
          <h1 className="text-2xl font-bold gradient-text">Create Account</h1>
          <p className="text-sm text-gray-500">Join RealTime Anomaly Defense</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="reg-email" className="text-xs text-gray-400 uppercase tracking-wider font-medium">Email</label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                id="reg-email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-gray-900 border border-gray-700 rounded-lg py-2.5 pl-9 pr-4 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="reg-password" className="text-xs text-gray-400 uppercase tracking-wider font-medium">Password</label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                id="reg-password"
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Min. 8 characters"
                className="w-full bg-gray-900 border border-gray-700 rounded-lg py-2.5 pl-9 pr-4 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
              />
            </div>
          </div>

          <button
            id="register-submit-btn"
            type="submit"
            disabled={loading}
            className="mt-2 flex items-center justify-center gap-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-60 text-white font-semibold py-2.5 rounded-lg transition-all duration-200 shadow-lg shadow-brand-500/20"
          >
            {loading ? (
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white" />
            ) : (
              <UserPlus size={16} />
            )}
            {loading ? 'Creating account…' : 'Create Account'}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600 mt-6">
          Already have an account?{' '}
          <Link to="/login" className="text-brand-400 hover:underline">Sign in</Link>
        </p>
      </div>
    </div>
  )
}
