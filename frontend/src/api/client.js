/**
 * Axios instance pre-configured with:
 *  - Base URL from VITE_API_BASE_URL env var (falls back to /api for the Vite proxy)
 *  - Automatic JWT Bearer header injection from localStorage
 *  - 401 response interceptor that clears stale tokens and redirects to /login
 */
import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
})

// ── Request interceptor — attach JWT ──────────────────────────────────────────
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// ── Response interceptor — handle 401 ────────────────────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default client
