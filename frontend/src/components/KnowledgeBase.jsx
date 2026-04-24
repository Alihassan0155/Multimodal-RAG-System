import { useState, useEffect } from 'react'
import { Database, FileText, Trash2, RefreshCw, AlertTriangle, Layers } from 'lucide-react'
import toast from 'react-hot-toast'

export default function KnowledgeBase() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [deleting, setDeleting] = useState(null)

  const fetchStats = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/stats')
      if (!res.ok) throw new Error('Failed to fetch stats')
      const data = await res.json()
      setStats(data)
    } catch (err) {
      const isOffline = err instanceof TypeError
      setError(isOffline ? 'offline' : err.message)
      if (!isOffline) toast.error('Could not load knowledge base stats')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStats() }, [])

  const handleDelete = async (docId, source) => {
    if (!confirm(`Delete all chunks for "${source}"?`)) return
    setDeleting(docId)
    try {
      const res = await fetch(`/api/docs/${docId}`, { method: 'DELETE' })
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail) }
      toast.success(`Deleted ${source}`)
      fetchStats()
    } catch (err) {
      toast.error(err.message)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-5 border-b border-white/[0.06] bg-dark-800/30 backdrop-blur-sm flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-accent-light" /> Knowledge Base
          </h2>
          <p className="text-sm text-gray-500 mt-1">Manage your ingested documents</p>
        </div>
        <button onClick={fetchStats} disabled={loading} className="btn-ghost flex items-center gap-2 text-sm">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 text-accent-light animate-spin" />
          </div>
        ) : error === 'offline' ? (
          <OfflineState onRetry={fetchStats} />
        ) : !stats || stats.total_chunks === 0 ? (
          <EmptyKB />
        ) : (
          <div className="space-y-6 animate-fade-in">
            {/* Stats cards */}
            <div className="grid grid-cols-3 gap-4">
              <StatCard icon={<Layers className="w-5 h-5" />} label="Total Chunks" value={stats.total_chunks} color="accent" />
              <StatCard icon={<FileText className="w-5 h-5" />} label="Documents" value={stats.total_documents} color="cyan" />
              <StatCard icon={<Database className="w-5 h-5" />} label="Collections" value="1" color="purple" />
            </div>

            {/* Document list */}
            <div>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Ingested Documents</h3>
              <div className="space-y-3">
                {stats.documents.map((doc) => (
                  <div key={doc.doc_id} className="glass-card flex items-center justify-between group">
                    <div className="flex items-center gap-4">
                      <div className="w-11 h-11 rounded-xl bg-cyan/10 border border-cyan/20 flex items-center justify-center">
                        <FileText className="w-5 h-5 text-cyan" />
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-white">{doc.source}</p>
                        <div className="flex items-center gap-3 mt-1">
                          <span className="text-xs text-gray-500">{doc.chunks} chunks</span>
                          <span className="text-xs text-gray-600">·</span>
                          <span className="text-xs text-gray-500">{doc.pages?.length || 0} pages</span>
                          <span className="text-xs text-gray-600">·</span>
                          <span className="text-xs text-gray-500">{doc.types?.join(', ')}</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleDelete(doc.doc_id, doc.source)}
                      disabled={deleting === doc.doc_id}
                      className="opacity-0 group-hover:opacity-100 p-2 text-gray-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all"
                    >
                      {deleting === doc.doc_id
                        ? <RefreshCw className="w-4 h-4 animate-spin" />
                        : <Trash2 className="w-4 h-4" />}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ icon, label, value, color }) {
  const colors = {
    accent: 'from-accent/10 to-accent/5 border-accent/20 text-accent-light',
    cyan: 'from-cyan/10 to-cyan/5 border-cyan/20 text-cyan',
    purple: 'from-purple-500/10 to-purple-500/5 border-purple-500/20 text-purple-400',
  }
  return (
    <div className={`glass-card bg-gradient-to-br ${colors[color]} flex items-center gap-4`}>
      <div className={`w-10 h-10 rounded-xl bg-white/[0.05] flex items-center justify-center ${colors[color].split(' ').pop()}`}>{icon}</div>
      <div>
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-xs text-gray-500">{label}</p>
      </div>
    </div>
  )
}

function EmptyKB() {
  return (
    <div className="flex flex-col items-center justify-center py-20 animate-fade-in">
      <div className="w-20 h-20 rounded-2xl bg-dark-700 border border-white/10 flex items-center justify-center mb-6">
        <AlertTriangle className="w-9 h-9 text-gray-600" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">Knowledge Base is Empty</h3>
      <p className="text-gray-500 text-center max-w-md">
        No documents have been ingested yet. Go to the Upload section to add PDF files.
      </p>
    </div>
  )
}

function OfflineState({ onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 animate-fade-in">
      <div className="w-20 h-20 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mb-6">
        <AlertTriangle className="w-9 h-9 text-red-400" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">API Server Offline</h3>
      <p className="text-gray-500 text-center max-w-md mb-6">
        Cannot connect to the backend. Make sure it's running:
        <code className="block mt-2 text-xs bg-dark-700 px-3 py-2 rounded-lg text-gray-300">python api.py</code>
      </p>
      <button onClick={onRetry} className="btn-primary text-sm flex items-center gap-2">
        <RefreshCw className="w-4 h-4" /> Retry
      </button>
    </div>
  )
}
