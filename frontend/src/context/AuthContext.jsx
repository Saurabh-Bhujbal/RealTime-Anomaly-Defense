/**
 * AuthContext
 * ───────────
 * Provides: { user, token, login, logout, isLoading }
 *
 * login(email, password):
 *   - POSTs to /api/auth/login
 *   - Stores the access_token in localStorage
 *   - Fetches /api/auth/me and stores the user object
 *
 * logout():
 *   - Clears localStorage and resets state
 *
 * On mount the context tries to rehydrate from localStorage so that
 * a page refresh does not log the user out.
 */
import { createContext, useContext, useEffect, useState } from 'react'
import client from '../api/client'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]         = useState(null)
  const [token, setToken]       = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  // ── Rehydrate on mount ──────────────────────────────────────────────────────
  useEffect(() => {
    const stored = localStorage.getItem('access_token')
    if (stored) {
      setToken(stored)
      client.defaults.headers.common['Authorization'] = `Bearer ${stored}`
      client.get('/api/auth/me')
        .then((res) => setUser(res.data))
        .catch(() => {
          localStorage.removeItem('access_token')
          setToken(null)
        })
        .finally(() => setIsLoading(false))
    } else {
      setIsLoading(false)
    }
  }, [])

  // ── Login ───────────────────────────────────────────────────────────────────
  const login = async (email, password) => {
    const { data } = await client.post('/api/auth/login', { email, password })
    const { access_token } = data
    localStorage.setItem('access_token', access_token)
    setToken(access_token)
    client.defaults.headers.common['Authorization'] = `Bearer ${access_token}`

    const meRes = await client.get('/api/auth/me')
    setUser(meRes.data)
    return meRes.data
  }

  // ── Register ────────────────────────────────────────────────────────────────
  const register = async (email, password) => {
    const { data } = await client.post('/api/auth/register', { email, password })
    return data
  }

  // ── Logout ──────────────────────────────────────────────────────────────────
  const logout = () => {
    localStorage.removeItem('access_token')
    delete client.defaults.headers.common['Authorization']
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  )
}

// Custom hook for easy consumption
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
