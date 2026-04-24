import { MessageSquare, Upload, Database, Sparkles } from 'lucide-react'

const navItems = [
  { id: 'chat', label: 'Chat', icon: MessageSquare, description: 'Ask questions' },
  { id: 'upload', label: 'Upload', icon: Upload, description: 'Ingest PDFs' },
  { id: 'knowledge', label: 'Knowledge', icon: Database, description: 'Manage docs' },
]

export default function Sidebar({ activeView, setActiveView }) {
  return (
    <aside className="w-[260px] h-screen flex flex-col border-r border-white/[0.06] bg-dark-800/50 backdrop-blur-xl">
      {/* Logo */}
      <div className="p-6 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-cyan flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">RAG System</h1>
            <p className="text-xs text-gray-500 font-medium">Multimodal Intelligence</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const isActive = activeView === item.id
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group
                ${isActive
                  ? 'bg-accent/15 text-accent-light border border-accent/20'
                  : 'text-gray-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
                }`}
            >
              <Icon className={`w-5 h-5 transition-colors ${isActive ? 'text-accent-light' : 'text-gray-500 group-hover:text-gray-300'}`} />
              <div className="text-left">
                <p className={`text-sm font-medium ${isActive ? 'text-accent-light' : ''}`}>{item.label}</p>
                <p className="text-[11px] text-gray-600">{item.description}</p>
              </div>
            </button>
          )
        })}
      </nav>

      {/* Status */}
      <div className="p-4 border-t border-white/[0.06]">
        <div className="glass-subtle p-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
          <span className="text-xs text-gray-400">API Connected</span>
        </div>
      </div>
    </aside>
  )
}
