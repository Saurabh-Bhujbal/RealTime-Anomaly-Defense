/**
 * App.jsx — root router + nav shell
 *
 * Routes:
 *   /           → redirect to /dashboard
 *   /login      → LoginPage
 *   /register   → RegisterPage
 *   /dashboard  → DashboardPage  (public landing; upload/analyze require login)
 *   /analytics  → AnalyticsPage
 *   /history    → HistoryPage    (requires auth)
 */
import { Navigate, Route, Routes, NavLink, useNavigate } from 'react-router-dom'
import { Shield, LayoutDashboard, BarChart2, History, LogOut, LogIn } from 'lucide-react'
import { useAuth } from './context/AuthContext'

import LoginPage    from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import DashboardPage  from './pages/DashboardPage'
import AnalyticsPage  from './pages/AnalyticsPage'
import HistoryPage    from './pages/HistoryPage'
import RequireAuth    from './components/RequireAuth'

const NAV = [
  { to: '/dashboard', label: 'Dashboard',  Icon: LayoutDashboard },
  { to: '/analytics', label: 'Analytics',  Icon: BarChart2 },
  { to: '/history',   label: 'History',    Icon: History },
]

function NavBar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => { logout(); navigate('/login') }

  return (
    <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-gray-950/80 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4 flex items-center justify-between h-14">
        {/* Brand */}
        <NavLink to="/dashboard" className="flex items-center gap-2">
          <Shield size={20} className="text-brand-500" />
          <span className="font-bold text-sm gradient-text hidden sm:block">
            RealTime Anomaly Defense
          </span>
        </NavLink>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150
                 ${isActive
                   ? 'bg-brand-600/20 text-brand-400'
                   : 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.04]'}`
              }
            >
              <Icon size={14} />
              <span className="hidden sm:block">{label}</span>
            </NavLink>
          ))}
        </div>

        {/* Auth */}
        <div className="flex items-center gap-2">
          {user ? (
            <>
              <span className="hidden sm:block text-xs text-gray-600 font-mono truncate max-w-[120px]">
                {user.email}
              </span>
              <button
                id="logout-btn"
                onClick={handleLogout}
                className="flex items-center gap-1 px-3 py-1.5 text-xs text-gray-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
              >
                <LogOut size={14} />
                <span className="hidden sm:block">Logout</span>
              </button>
            </>
          ) : (
            <NavLink
              to="/login"
              className="flex items-center gap-1 px-3 py-1.5 text-xs text-brand-400 border border-brand-500/40 rounded-lg hover:bg-brand-500/10 transition-all"
            >
              <LogIn size={14} />
              Sign In
            </NavLink>
          )}
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  const { isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <span className="animate-spin rounded-full h-8 w-8 border-2 border-brand-500/30 border-t-brand-500" />
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <NavBar />
      <Routes>
        <Route path="/"           element={<Navigate to="/dashboard" replace />} />
        <Route path="/login"      element={<LoginPage />} />
        <Route path="/register"   element={<RegisterPage />} />
        <Route path="/dashboard"  element={<DashboardPage />} />
        <Route path="/analytics"  element={<AnalyticsPage />} />
        <Route path="/history"    element={<RequireAuth><HistoryPage /></RequireAuth>} />
      </Routes>
    </div>
  )
}
