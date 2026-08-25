import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { Shield, Mail, Lock, LogIn } from 'lucide-react'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate   = useNavigate()
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      await login(email, password)
      toast.success('Welcome back!')
      navigate('/dashboard')
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Login failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-96 h-96 bg-brand-600/20 rounded-full blur-3xl" />
      </div>

      <div className="glass w-full max-w-md p-8 z-10">
        {/* Header */}
        <div className="flex flex-col items-center gap-2 mb-8">
          <div className="p-3 bg-brand-600/20 rounded-xl">
            <Shield size={28} className="text-brand-400" />
          </div>
          <h1 className="text-2xl font-bold gradient-text">RealTime Anomaly Defense</h1>
          <p className="text-sm text-gray-500">Sign in to your account</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="email" className="text-xs text-gray-400 uppercase tracking-wider font-medium">
              Email
            </label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                id="email"
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
            <label htmlFor="password" className="text-xs text-gray-400 uppercase tracking-wider font-medium">
              Password
            </label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                id="password"
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-gray-900 border border-gray-700 rounded-lg py-2.5 pl-9 pr-4 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
              />
            </div>
          </div>

          <button
            id="login-submit-btn"
            type="submit"
            disabled={loading}
            className="mt-2 flex items-center justify-center gap-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-60 text-white font-semibold py-2.5 rounded-lg transition-all duration-200 shadow-lg shadow-brand-500/20"
          >
            {loading ? (
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-white/30 border-t-white" />
            ) : (
              <LogIn size={16} />
            )}
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600 mt-6">
          No account?{' '}
          <Link to="/register" className="text-brand-400 hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  )
}
