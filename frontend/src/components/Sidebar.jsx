import { MessageSquare, Upload, Library } from 'lucide-react'

const navItems = [
  { id: 'chat',      label: 'Chat',         icon: MessageSquare, description: 'Ask your documents' },
  { id: 'upload',    label: 'Add Document', icon: Upload,        description: 'Upload a PDF' },
  { id: 'knowledge', label: 'My Library',   icon: Library,       description: 'View your files' },
]

function DocFlowMark() {
  return (
    <svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
      <defs>
        <linearGradient id="df-grad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6c5ce7" />
          <stop offset="100%" stopColor="#00cec9" />
        </linearGradient>
      </defs>
      {/* App icon background */}
      <rect width="36" height="36" rx="9" fill="url(#df-grad)" />
      {/* Document body */}
      <path d="M9 7h11l7 7v15a2 2 0 01-2 2H9a2 2 0 01-2-2V9a2 2 0 012-2z" fill="white" opacity="0.95" />
      {/* Folded corner */}
      <path d="M20 7v7h7" fill="none" stroke="white" strokeWidth="1.4" strokeLinejoin="round" opacity="0.5" />
      {/* Content lines — simulate text in doc */}
      <rect x="11" y="17" width="11" height="1.8" rx="0.9" fill="#6c5ce7" opacity="0.7" />
      <rect x="11" y="20.5" width="7.5" height="1.8" rx="0.9" fill="#6c5ce7" opacity="0.5" />
      {/* Flow arrow — the "Flow" in DocFlow */}
      <rect x="11" y="24" width="9" height="1.8" rx="0.9" fill="#00cec9" opacity="0.6" />
    </svg>
  )
}

export default function Sidebar({ activeView, setActiveView }) {
  return (
    <aside className="w-[240px] h-screen flex flex-col border-r border-white/[0.06] bg-dark-800/50 backdrop-blur-xl">

      {/* Logo */}
      <div className="px-5 py-5 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 shrink-0">
            <DocFlowMark />
          </div>
          <div className="leading-none">
            <h1 className="text-[17px] font-bold text-white tracking-tight">DocFlow</h1>
            <p className="text-[11px] text-gray-500 mt-0.5 font-medium">Chat with your documents</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-0.5">
        {navItems.map((item) => {
          const isActive = activeView === item.id
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={() => setActiveView(item.id)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all duration-150 group
                ${isActive
                  ? 'bg-accent/15 border border-accent/25'
                  : 'border border-transparent hover:bg-white/[0.04] hover:border-white/[0.05]'
                }`}
            >
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors
                ${isActive ? 'bg-accent/20' : 'bg-white/[0.04] group-hover:bg-white/[0.07]'}`}>
                <Icon className={`w-4 h-4 ${isActive ? 'text-accent-light' : 'text-gray-500 group-hover:text-gray-300'}`} />
              </div>
              <div className="text-left min-w-0">
                <p className={`text-sm font-medium truncate ${isActive ? 'text-accent-light' : 'text-gray-400 group-hover:text-white'}`}>
                  {item.label}
                </p>
                <p className="text-[11px] text-gray-600 truncate">{item.description}</p>
              </div>
            </button>
          )
        })}
      </nav>

      {/* Status pill */}
      <div className="p-3 border-t border-white/[0.06]">
        <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-white/[0.02] border border-white/[0.05]">
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-50" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          <span className="text-xs text-gray-400 font-medium">Connected</span>
        </div>
      </div>

    </aside>
  )
}
