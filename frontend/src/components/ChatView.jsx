import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles, User, ChevronDown, ChevronUp, FileText, Zap } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import toast from 'react-hot-toast'

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
        setDocCount(data.total_documents || 0)
      }
    } catch {
      // API not running — will handle gracefully when user tries to query
      setDocCount(null)
    }
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

    // Check for documents first (frontend-side guard)
    if (docCount === 0) {
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
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
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
          <Sparkles className="w-5 h-5 text-accent-light" />
          RAG Chat
        </h2>
        <p className="text-sm text-gray-500 mt-1">Ask questions about your ingested documents</p>
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
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-accent/20 to-cyan/20 border border-accent/20 flex items-center justify-center mb-6 glow-accent">
        <Sparkles className="w-9 h-9 text-accent-light" />
      </div>
      <h3 className="text-2xl font-bold text-white mb-2">Start a Conversation</h3>
      <p className="text-gray-500 mb-8 text-center max-w-md">
        Ask questions about your ingested documents. The RAG pipeline will retrieve relevant context and generate grounded answers.
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
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-cyan flex items-center justify-center shrink-0 mt-1">
          <Sparkles className="w-4 h-4 text-white" />
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

        {/* Pipeline metadata for assistant messages */}
        {!isUser && !message.isError && message.intent && (
          <PipelineInfo message={message} />
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

  const intentColors = {
    lookup: 'text-blue-400 bg-blue-400/10 border-blue-400/20',
    aggregation: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
    comparison: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
    boolean: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  }

  return (
    <div className="mt-2 ml-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        <Zap className="w-3 h-3" />
        <span className={`px-2 py-0.5 rounded-md text-[11px] font-medium border ${intentColors[message.intent] || 'text-gray-400 bg-gray-400/10 border-gray-400/20'}`}>
          {message.intent?.toUpperCase()}
        </span>
        <span>· {message.sources?.length || 0} sources · {(message.confidence * 100).toFixed(0)}% conf</span>
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {expanded && (
        <div className="mt-3 space-y-3 animate-slide-up">
          {/* Expanded queries */}
          {message.expanded_queries?.length > 1 && (
            <div className="glass-subtle p-3">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Query Expansion</p>
              <div className="space-y-1">
                {message.expanded_queries.map((q, i) => (
                  <p key={i} className="text-xs text-gray-400">
                    <span className="text-accent-light/60 mr-1">{i === 0 ? '●' : '○'}</span> {q}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Sources */}
          {message.sources?.length > 0 && (
            <div className="glass-subtle p-3">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Sources</p>
              <div className="space-y-2">
                {message.sources.map((s, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <FileText className="w-3.5 h-3.5 text-cyan mt-0.5 shrink-0" />
                    <div>
                      <span className="text-gray-300 font-medium">{s.source}</span>
                      <span className="text-gray-500"> · page {s.page}</span>
                      <span className="text-gray-600"> · {s.type}</span>
                      <span className={`ml-1 ${s.rerank_score >= 3 ? 'text-emerald-400' : s.rerank_score >= 0 ? 'text-amber-400' : 'text-red-400'}`}>
                        ({s.rerank_score?.toFixed(2)})
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}


/* ── Typing indicator ─────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex gap-4 animate-fade-in">
      <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-cyan flex items-center justify-center shrink-0">
        <Sparkles className="w-4 h-4 text-white" />
      </div>
      <div className="glass rounded-2xl px-5 py-4">
        <div className="flex gap-1.5 items-center h-5">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
          <span className="text-xs text-gray-500 ml-2">Processing pipeline...</span>
        </div>
      </div>
    </div>
  )
}
