import { useState } from 'react'
import './index.css'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import UploadView from './components/UploadView'
import KnowledgeBase from './components/KnowledgeBase'
import { Toaster } from 'react-hot-toast'

function App() {
  const [activeView, setActiveView] = useState('chat')

  const renderView = () => {
    switch (activeView) {
      case 'chat':
        return <ChatView />
      case 'upload':
        return <UploadView />
      case 'knowledge':
        return <KnowledgeBase />
      default:
        return <ChatView />
    }
  }

  return (
    <div className="flex h-screen bg-dark-900 bg-mesh">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'rgba(26, 26, 46, 0.95)',
            color: '#e2e8f0',
            border: '1px solid rgba(255,255,255,0.1)',
            backdropFilter: 'blur(20px)',
            borderRadius: '12px',
          },
          success: { iconTheme: { primary: '#00cec9', secondary: '#0a0a0f' } },
          error: { iconTheme: { primary: '#ff6b6b', secondary: '#0a0a0f' } },
        }}
      />
      <Sidebar activeView={activeView} setActiveView={setActiveView} />
      <main className="flex-1 overflow-hidden">
        {renderView()}
      </main>
    </div>
  )
}

export default App
