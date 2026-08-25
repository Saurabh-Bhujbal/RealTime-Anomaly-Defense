/**
 * RequireAuth — client-side route guard.
 * ───────────────────────────────────────
 * Wrap any route element that must not be reachable without a logged-in user.
 * If there is no authenticated user, redirect to /login (preserving the
 * attempted location so the user can be returned there after signing in).
 *
 * Used to protect the /history route. The Dashboard itself loads publicly
 * (it is the app's landing page); its analysis actions gate login inline —
 * see DashboardPage — so it is intentionally NOT wrapped here.
 *
 * App.jsx already blocks rendering until AuthContext finishes its mount-time
 * rehydration (isLoading), but we guard on isLoading here too so the component
 * is safe to use anywhere.
 */
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function RequireAuth({ children }) {
  const { user, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center">
        <span className="animate-spin rounded-full h-8 w-8 border-2 border-brand-500/30 border-t-brand-500" />
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return children
}
