import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

export default function UploadView() {
  const [uploadState, setUploadState] = useState('idle')
  const [uploadInfo, setUploadInfo]   = useState(null)
  const [progress, setProgress]       = useState(0)
  const [stage, setStage]             = useState('')

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      toast.error('Only PDF files are accepted')
      return
    }

    setUploadState('uploading')
    setUploadInfo({ name: file.name, size: (file.size / 1024 / 1024).toFixed(2) })
    setProgress(0)
    setStage('Uploading…')

    const stages = [
      { at: 15,  label: 'Reading your PDF…' },
      { at: 35,  label: 'Scanning for tables…' },
      { at: 55,  label: 'Organizing sections…' },
      { at: 75,  label: 'Making content searchable…' },
      { at: 88,  label: 'Processing images…' },
    ]
    let stageIdx = 0
    let p = 0
    const timer = setInterval(() => {
      p = Math.min(p + Math.random() * 6 + 2, 88)
      setProgress(Math.round(p))
      if (stageIdx < stages.length && p >= stages[stageIdx].at) {
        setStage(stages[stageIdx].label)
        stageIdx++
      }
    }, 400)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/ingest', { method: 'POST', body: formData })
      clearInterval(timer)

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }

      setProgress(100)
      setStage('Ready!')
      setUploadState('success')
      setUploadInfo(prev => ({ ...prev, message: 'Added to your library' }))
      toast.success(`${file.name} is ready to chat!`)
    } catch (err) {
      clearInterval(timer)
      setUploadState('error')
      setUploadInfo(prev => ({ ...prev, error: err.message }))
      toast.error(err.message)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: uploadState === 'uploading',
  })

  const reset = () => { setUploadState('idle'); setUploadInfo(null); setProgress(0); setStage('') }

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-5 border-b border-white/[0.06] bg-dark-800/30 backdrop-blur-sm">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Upload className="w-5 h-5 text-cyan" /> Add a Document
        </h2>
        <p className="text-sm text-gray-500 mt-1">Upload a PDF and start chatting with it instantly</p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        <div className="max-w-2xl mx-auto space-y-6">

          {/* Drop zone */}
          <div
            {...getRootProps()}
            className={`relative rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300 cursor-pointer group
              ${isDragActive       ? 'border-accent bg-accent/5 scale-[1.02]'
              : uploadState === 'uploading' ? 'border-amber-500/30 bg-amber-500/5 cursor-wait'
              : uploadState === 'success'   ? 'border-emerald-500/30 bg-emerald-500/5'
              : uploadState === 'error'     ? 'border-red-500/30 bg-red-500/5'
              : 'border-white/10 hover:border-accent/40 hover:bg-white/[0.02]'}`}
          >
            <input {...getInputProps()} />

            {uploadState === 'idle' && (
              <div className="animate-fade-in">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan/10 to-accent/10 border border-white/10 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <Upload className="w-9 h-9 text-cyan/70 group-hover:text-cyan transition-colors" />
                </div>
                <p className="text-lg font-semibold text-white mb-2">
                  {isDragActive ? 'Drop your PDF here' : 'Drag & drop your PDF here'}
                </p>
                <p className="text-sm text-gray-500">
                  or <span className="text-accent-light underline underline-offset-2">browse files</span>
                </p>
                <p className="text-xs text-gray-600 mt-4">PDF files only · Any size</p>
              </div>
            )}

            {uploadState === 'uploading' && (
              <div className="animate-fade-in">
                <Loader2 className="w-12 h-12 text-amber-400 mx-auto mb-4 animate-spin" />
                <p className="text-lg font-semibold text-white mb-1">Processing your document…</p>
                <p className="text-sm text-gray-400 mb-6 truncate max-w-xs mx-auto">{uploadInfo?.name}</p>

                {/* Progress bar */}
                <div className="w-full max-w-xs mx-auto">
                  <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-accent to-cyan rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between mt-2">
                    <p className="text-xs text-gray-400">{stage}</p>
                    <p className="text-xs text-gray-500 tabular-nums">{progress}%</p>
                  </div>
                </div>
              </div>
            )}

            {uploadState === 'success' && (
              <div className="animate-slide-up">
                <CheckCircle className="w-14 h-14 text-emerald-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-1">Ready to chat!</p>
                <p className="text-sm text-gray-400 mb-1">{uploadInfo?.name}</p>
                <p className="text-xs text-emerald-400/80">{uploadInfo?.message}</p>
                <button
                  onClick={(e) => { e.stopPropagation(); reset() }}
                  className="btn-primary mt-6 text-sm"
                >
                  Add Another Document
                </button>
              </div>
            )}

            {uploadState === 'error' && (
              <div className="animate-slide-up">
                <XCircle className="w-14 h-14 text-red-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-1">Something went wrong</p>
                <p className="text-sm text-red-400/80 mb-1">{uploadInfo?.error}</p>
                <p className="text-xs text-gray-500">Make sure the file is a valid PDF and try again</p>
                <button
                  onClick={(e) => { e.stopPropagation(); reset() }}
                  className="btn-primary mt-6 text-sm"
                >
                  Try Again
                </button>
              </div>
            )}
          </div>

          {/* What happens — shown only when idle */}
          {uploadState === 'idle' && (
            <div className="glass-card animate-fade-in">
              <p className="text-sm font-semibold text-gray-300 mb-4">What happens after you upload?</p>
              <div className="space-y-3">
                {[
                  { emoji: '📖', label: 'We read your document',        desc: 'Text, tables, and images are extracted automatically' },
                  { emoji: '🧩', label: 'We organize the content',      desc: 'Your document is split into meaningful sections' },
                  { emoji: '🔍', label: 'We make it searchable',        desc: 'Every section gets indexed so nothing is missed' },
                  { emoji: '💬', label: 'You start chatting',           desc: 'Ask any question and get answers from your document' },
                ].map((step) => (
                  <div key={step.label} className="flex items-start gap-3">
                    <span className="text-xl leading-none mt-0.5">{step.emoji}</span>
                    <div>
                      <p className="text-sm font-medium text-white">{step.label}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
