import { useState, useEffect, useMemo } from 'react'
import { FileText, Trash2, RefreshCw, Library, WifiOff, Search, X, Upload, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'

/* ── API ─────────────────────────────────────────────────────── */
export default function KnowledgeBase({ onNavigate }) {
  const [stats, setStats]       = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [deleting, setDeleting] = useState(null)   // doc_id currently being deleted
  const [confirm, setConfirm]   = useState(null)   // doc_id awaiting inline confirmation
  const [search, setSearch]     = useState('')

  const fetchStats = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/stats')
      if (!res.ok) throw new Error('Failed to fetch stats')
      setStats(await res.json())
    } catch (err) {
      setError(err instanceof TypeError ? 'offline' : err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStats() }, [])

  const handleDelete = async (docId, source) => {
    setDeleting(docId)
    setConfirm(null)
    try {
      const res = await fetch(`/api/docs/${docId}`, { method: 'DELETE' })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail) }
      toast.success(`"${source}" removed from your library`)
      fetchStats()
    } catch (err) {
      toast.error(err.message)
    } finally {
      setDeleting(null)
    }
  }

  const filteredDocs = useMemo(() => {
    if (!stats?.documents) return []
    const q = search.trim().toLowerCase()
    return q ? stats.documents.filter(d => d.source.toLowerCase().includes(q)) : stats.documents
  }, [stats, search])

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="flex flex-col h-full">

      {/* Header */}
      <header className="px-8 py-5 border-b border-white/[0.06] bg-dark-800/30 backdrop-blur-sm flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Library className="w-5 h-5 text-accent-light shrink-0" />
            My Library
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            {stats && !loading
              ? `${stats.total_documents} ${stats.total_documents === 1 ? 'document' : 'documents'} ready to chat`
              : 'All documents you can chat with'}
          </p>
        </div>
        <button
          onClick={fetchStats}
          disabled={loading}
          className="btn-ghost flex items-center gap-2 text-sm shrink-0"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </header>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        {loading ? (
          <SkeletonList />
        ) : error === 'offline' ? (
          <OfflineState onRetry={fetchStats} />
        ) : !stats || stats.total_documents === 0 ? (
          <EmptyLibrary />
        ) : (
          <div className="max-w-2xl mx-auto space-y-5 animate-fade-in">

            {/* Search bar */}
            <div className="relative">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none" />
              <input
                type="text"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search documents…"
                className="input-field pl-10 pr-10 text-sm"
              />
              {search && (
                <button
                  onClick={() => setSearch('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Document list */}
            {filteredDocs.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <Search className="w-8 h-8 mx-auto mb-3 opacity-40" />
                <p className="text-sm">No documents match "{search}"</p>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredDocs.map((doc) => (
                  <DocCard
                    key={doc.doc_id}
                    doc={doc}
                    isDeleting={deleting === doc.doc_id}
                    confirmPending={confirm === doc.doc_id}
                    onAskConfirm={() => setConfirm(doc.doc_id)}
                    onCancelConfirm={() => setConfirm(null)}
                    onConfirmDelete={() => handleDelete(doc.doc_id, doc.source)}
                  />
                ))}
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  )
}

/* ── Document card ───────────────────────────────────────────── */
function DocCard({ doc, isDeleting, confirmPending, onAskConfirm, onCancelConfirm, onConfirmDelete }) {
  const pageCount = doc.pages?.length ?? 0

  return (
    <div className={`relative group rounded-2xl border transition-all duration-200 overflow-hidden
      ${confirmPending
        ? 'border-red-500/30 bg-red-500/5'
        : 'border-white/[0.07] bg-white/[0.02] hover:bg-white/[0.04] hover:border-white/[0.12]'
      }`}
    >
      {/* Left accent bar */}
      <div className={`absolute left-0 top-0 bottom-0 w-[3px] rounded-l-2xl transition-colors duration-200
        ${confirmPending ? 'bg-red-500/60' : 'bg-gradient-to-b from-accent to-cyan opacity-0 group-hover:opacity-100'}`}
      />

      <div className="px-5 py-4 flex items-center gap-4">
        {/* Icon */}
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 transition-colors
          ${confirmPending ? 'bg-red-500/10 border border-red-500/20' : 'bg-accent/8 border border-accent/15'}`}>
          <FileText className={`w-5 h-5 ${confirmPending ? 'text-red-400' : 'text-accent-light'}`} />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-white truncate leading-snug" title={doc.source}>
            {doc.source}
          </p>
          {!confirmPending && (
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[11px] font-medium px-1.5 py-0.5 rounded-md bg-accent/10 text-accent-light">
                PDF
              </span>
              {pageCount > 0 && (
                <span className="text-[11px] text-gray-500">
                  {pageCount} {pageCount === 1 ? 'page' : 'pages'}
                </span>
              )}
              <span className="flex items-center gap-1 text-[11px] text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                Ready
              </span>
            </div>
          )}
          {confirmPending && (
            <p className="text-xs text-red-400 mt-0.5">
              This will remove all indexed content for this file.
            </p>
          )}
        </div>

        {/* Actions */}
        {isDeleting ? (
          <RefreshCw className="w-4 h-4 text-gray-500 animate-spin shrink-0" />
        ) : confirmPending ? (
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={onCancelConfirm}
              className="text-xs px-3 py-1.5 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:border-white/20 transition-all"
            >
              Cancel
            </button>
            <button
              onClick={onConfirmDelete}
              className="text-xs px-3 py-1.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-400 hover:bg-red-500/25 transition-all font-medium"
            >
              Remove
            </button>
          </div>
        ) : (
          <button
            onClick={onAskConfirm}
            title="Remove from library"
            className="opacity-0 group-hover:opacity-100 p-2 text-gray-600 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all shrink-0"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}

/* ── Skeleton ────────────────────────────────────────────────── */
function SkeletonList() {
  return (
    <div className="max-w-2xl mx-auto space-y-5 animate-pulse">
      {/* Search skeleton */}
      <div className="h-11 rounded-xl bg-white/[0.04]" />
      {/* Card skeletons */}
      {[1, 2, 3].map(i => (
        <div key={i} className="rounded-2xl border border-white/[0.06] bg-white/[0.02] px-5 py-4 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-white/[0.05] shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-3.5 rounded-md bg-white/[0.07] w-3/5" />
            <div className="h-2.5 rounded-md bg-white/[0.04] w-1/3" />
          </div>
        </div>
      ))}
    </div>
  )
}

/* ── Empty state ─────────────────────────────────────────────── */
function EmptyLibrary() {
  return (
    <div className="flex flex-col items-center justify-center py-24 animate-fade-in">
      <div className="w-16 h-16 rounded-2xl bg-accent/8 border border-accent/15 flex items-center justify-center mb-5">
        <Library className="w-7 h-7 text-accent-light opacity-70" />
      </div>
      <h3 className="text-lg font-bold text-white mb-2">Your library is empty</h3>
      <p className="text-gray-500 text-center text-sm max-w-xs leading-relaxed">
        Upload a PDF to get started — then ask it anything.
      </p>
    </div>
  )
}

/* ── Offline state ───────────────────────────────────────────── */
function OfflineState({ onRetry }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 animate-fade-in">
      <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mb-5">
        <WifiOff className="w-7 h-7 text-red-400 opacity-80" />
      </div>
      <h3 className="text-lg font-bold text-white mb-2">Can't connect</h3>
      <p className="text-gray-500 text-center text-sm max-w-xs leading-relaxed mb-6">
        DocFlow's backend isn't reachable. Make sure the server is running, then try again.
      </p>
      <button onClick={onRetry} className="btn-primary text-sm flex items-center gap-2">
        <RefreshCw className="w-4 h-4" /> Try again
      </button>
    </div>
  )
}
