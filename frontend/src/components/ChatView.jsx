import { useState, useRef, useEffect } from 'react'
import { Send, User, ChevronDown, ChevronUp, FileText, Maximize2, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import toast from 'react-hot-toast'

/* ── DocFlow brand mark — used wherever the AI identity is shown ── */
function DocFlowMark({ className = 'w-full h-full' }) {
  return (
    <svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
      <defs>
        <linearGradient id="df-chat-grad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6c5ce7" />
          <stop offset="100%" stopColor="#00cec9" />
        </linearGradient>
      </defs>
      <rect width="36" height="36" rx="9" fill="url(#df-chat-grad)" />
      <path d="M9 7h11l7 7v15a2 2 0 01-2 2H9a2 2 0 01-2-2V9a2 2 0 012-2z" fill="white" opacity="0.95" />
      <path d="M20 7v7h7" fill="none" stroke="white" strokeWidth="1.4" strokeLinejoin="round" opacity="0.5" />
      <rect x="11" y="17" width="11" height="1.8" rx="0.9" fill="#6c5ce7" opacity="0.7" />
      <rect x="11" y="20.5" width="7.5" height="1.8" rx="0.9" fill="#6c5ce7" opacity="0.5" />
      <rect x="11" y="24" width="9" height="1.8" rx="0.9" fill="#00cec9" opacity="0.6" />
    </svg>
  )
}

export default function ChatView() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [docCount, setDocCount] = useState(null) // null = unknown, 0 = empty, >0 = has docs
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
    // Check if any documents are ingested
    checkDocuments()
  }, [])

  const checkDocuments = async () => {
    try {
      const res = await fetch('/api/stats')
      if (res.ok) {
        const data = await res.json()
        const count = data.total_documents || 0
        setDocCount(count)
        return count
      }
    } catch {
      setDocCount(null)
    }
    return null
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || isLoading) return

    // Add user message
    const userMsg = { role: 'user', content: question, timestamp: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    // Always fetch a fresh count — stale mount-time value can be wrong
    // if the user added documents after navigating to this view.
    const freshCount = await checkDocuments()
    if (freshCount === 0) {
      const noDocsMsg = {
        role: 'assistant',
        content: '📄 **No documents in the knowledge base yet.**\n\nPlease go to the **Upload** section and ingest a PDF first — then come back and ask your question!',
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, noDocsMsg])
      setIsLoading(false)
      return
    }

    try {
      // Send last 6 messages (3 Q&A turns) as conversation history
      const history = messages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .filter(m => !m.isError)
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }))

      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Query failed')
      }

      const data = await res.json()

      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        intent: data.intent,
        confidence: data.confidence,
        sources: data.sources || [],
        expanded_queries: data.expanded_queries || [],
        images: data.images || [],
        timestamp: Date.now(),
      }

      setMessages(prev => [...prev, assistantMsg])
      // Refresh doc count after successful query
      checkDocuments()
    } catch (err) {
      const isConnectionError = err.message?.includes('fetch') || err.message?.includes('NetworkError') || err.name === 'TypeError'
      const errorContent = isConnectionError
        ? '⚠️ **Cannot connect to the API server.**\n\nMake sure the backend is running:\n```\npython api.py\n```'
        : `**Error:** ${err.message}`

      toast.error(isConnectionError ? 'API server not reachable' : err.message)
      const errorMsg = {
        role: 'assistant',
        content: errorContent,
        timestamp: Date.now(),
        isError: true,
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="px-8 py-5 border-b border-white/[0.06] bg-dark-800/30 backdrop-blur-sm">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="w-6 h-6 shrink-0"><DocFlowMark /></span>
          DocFlow Chat
        </h2>
        <p className="text-sm text-gray-500 mt-1">Ask anything about your documents</p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
        {messages.length === 0 && (
          <EmptyState onExampleClick={(q) => { setInput(q); inputRef.current?.focus() }} />
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {isLoading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-8 py-5 border-t border-white/[0.06] bg-dark-800/30 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            className="input-field flex-1"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="btn-primary flex items-center gap-2 shrink-0"
          >
            <Send className="w-4 h-4" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </form>
      </div>
    </div>
  )
}


/* ── Empty state with example queries ─────────────────────── */
function EmptyState({ onExampleClick }) {
  const examples = [
    "What are the key findings of this paper?",
    "List all methods described in the document",
    "Does the system use neural networks?",
  ]

  return (
    <div className="flex-1 flex flex-col items-center justify-center py-20 animate-fade-in">
      <div className="w-20 h-20 mb-6 glow-accent">
        <DocFlowMark />
      </div>
      <h3 className="text-2xl font-bold text-white mb-2">Ask your documents anything</h3>
      <p className="text-gray-500 mb-8 text-center max-w-md">
        Upload a PDF and DocFlow will find the answers — summaries, details, comparisons, and more.
      </p>
      <div className="space-y-3 w-full max-w-lg">
        {examples.map((q, i) => (
          <button
            key={i}
            onClick={() => onExampleClick(q)}
            className="w-full text-left glass-subtle px-5 py-3.5 text-gray-300 hover:text-white hover:bg-white/[0.06] hover:border-accent/20 transition-all duration-200 group"
          >
            <span className="text-accent-light mr-2 opacity-50 group-hover:opacity-100">→</span>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}


/* ── Message bubble ───────────────────────────────────────── */
function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-4 animate-slide-up ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-9 h-9 shrink-0 mt-1">
          <DocFlowMark />
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? 'order-first' : ''}`}>
        <div className={`rounded-2xl px-5 py-4 ${
          isUser
            ? 'bg-accent/20 border border-accent/30 text-gray-100'
            : message.isError
              ? 'glass border-red-500/30'
              : 'glass'
        }`}>
          {isUser ? (
            <p className="text-[15px] leading-relaxed">{message.content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Sources */}
        {!isUser && !message.isError && message.intent && (
          <PipelineInfo message={message} />
        )}

        {/* Images */}
        {!isUser && !message.isError && message.images?.length > 0 && (
          <ImagePanel images={message.images} />
        )}
      </div>

      {isUser && (
        <div className="w-9 h-9 rounded-xl bg-dark-600 border border-white/10 flex items-center justify-center shrink-0 mt-1">
          <User className="w-4 h-4 text-gray-400" />
        </div>
      )}
    </div>
  )
}


/* ── Pipeline info expandable ─────────────────────────────── */
function PipelineInfo({ message }) {
  const [expanded, setExpanded] = useState(false)
  const count = message.sources?.length || 0
  if (count === 0) return null

  return (
    <div className="mt-2 ml-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        <FileText className="w-3 h-3" />
        <span>{count} source{count !== 1 ? 's' : ''}</span>
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {expanded && (
        <div className="mt-2 glass-subtle p-3 animate-slide-up space-y-1.5">
          {message.sources.map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <FileText className="w-3 h-3 text-cyan shrink-0" />
              <span className="text-gray-300 font-medium">{s.source}</span>
              <span className="text-gray-500">· p.{s.page}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


/* ── Image panel with lightbox ────────────────────────────── */
function ImagePanel({ images }) {
  const [lightbox, setLightbox] = useState(null)

  useEffect(() => {
    if (!lightbox) return
    const onKey = (e) => { if (e.key === 'Escape') setLightbox(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightbox])

  return (
    <>
      <div className="mt-3 space-y-2">
        <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider ml-1">
          Visual References
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {images.map((img) => (
            <button
              key={img.image_id}
              onClick={() => setLightbox(img)}
              className="glass-subtle rounded-xl overflow-hidden text-left group hover:border-accent/40 transition-all duration-200"
            >
              <div className="relative bg-white/[0.03] flex items-center justify-center min-h-[120px]">
                <img
                  src={`http://localhost:8000${img.url}`}
                  alt={img.caption || img.image_id}
                  className="w-full object-contain max-h-52"
                  onError={(e) => { e.currentTarget.closest('button').style.display = 'none' }}
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                  <Maximize2 className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
                </div>
              </div>
              {(img.caption || img.source) && (
                <div className="px-3 py-2 space-y-0.5 border-t border-white/[0.05]">
                  {img.caption && (
                    <p className="text-xs text-gray-300 leading-snug line-clamp-2">{img.caption}</p>
                  )}
                  <p className="text-[11px] text-gray-500">{img.source} · p.{img.page}</p>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Lightbox overlay */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/75 backdrop-blur-sm animate-fade-in"
          onClick={() => setLightbox(null)}
        >
          <div
            className="relative max-w-4xl w-full glass rounded-2xl overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setLightbox(null)}
              className="absolute top-3 right-3 z-10 p-1.5 rounded-lg bg-dark-700/80 text-gray-400 hover:text-white hover:bg-dark-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="bg-white/[0.03] flex items-center justify-center p-4">
              <img
                src={`http://localhost:8000${lightbox.url}`}
                alt={lightbox.caption || lightbox.image_id}
                className="max-w-full max-h-[75vh] object-contain"
              />
            </div>
            {(lightbox.caption || lightbox.source) && (
              <div className="px-5 py-3 border-t border-white/[0.06]">
                {lightbox.caption && (
                  <p className="text-sm text-gray-300 leading-snug">{lightbox.caption}</p>
                )}
                <p className="text-xs text-gray-500 mt-0.5">{lightbox.source} · p.{lightbox.page}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}


/* ── Typing indicator ─────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex gap-4 animate-fade-in">
      <div className="w-9 h-9 shrink-0">
        <DocFlowMark />
      </div>
      <div className="glass rounded-2xl px-5 py-4">
        <div className="flex gap-1.5 items-center h-5">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
          <span className="text-xs text-gray-500 ml-2">Thinking…</span>
        </div>
      </div>
    </div>
  )
}
