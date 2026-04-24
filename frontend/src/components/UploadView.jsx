import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

export default function UploadView() {
  const [uploadState, setUploadState] = useState('idle')
  const [uploadInfo, setUploadInfo] = useState(null)
  const [progress, setProgress] = useState(0)

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

    const progressInterval = setInterval(() => {
      setProgress(prev => prev >= 90 ? (clearInterval(progressInterval), 90) : prev + Math.random() * 15)
    }, 500)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch('/api/ingest', { method: 'POST', body: formData })
      clearInterval(progressInterval)
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Upload failed') }
      const data = await res.json()
      setProgress(100)
      setUploadState('success')
      setUploadInfo(prev => ({ ...prev, message: data.message }))
      toast.success(`${file.name} ingested successfully!`)
    } catch (err) {
      clearInterval(progressInterval)
      setUploadState('error')
      setUploadInfo(prev => ({ ...prev, error: err.message }))
      toast.error(err.message)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'application/pdf': ['.pdf'] }, multiple: false, disabled: uploadState === 'uploading',
  })

  const reset = () => { setUploadState('idle'); setUploadInfo(null); setProgress(0) }

  return (
    <div className="flex flex-col h-full">
      <header className="px-8 py-5 border-b border-white/[0.06] bg-dark-800/30 backdrop-blur-sm">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Upload className="w-5 h-5 text-cyan" /> Upload Document
        </h2>
        <p className="text-sm text-gray-500 mt-1">Ingest PDF files into the knowledge base</p>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        <div className="max-w-2xl mx-auto">
          <div {...getRootProps()} className={`relative rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300 cursor-pointer group
            ${isDragActive ? 'border-accent bg-accent/5 scale-[1.02]'
              : uploadState === 'uploading' ? 'border-amber-500/30 bg-amber-500/5 cursor-wait'
              : uploadState === 'success' ? 'border-emerald-500/30 bg-emerald-500/5'
              : uploadState === 'error' ? 'border-red-500/30 bg-red-500/5'
              : 'border-white/10 hover:border-accent/40 hover:bg-white/[0.02]'}`}>
            <input {...getInputProps()} />

            {uploadState === 'idle' && (
              <div className="animate-fade-in">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan/10 to-accent/10 border border-white/10 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <Upload className="w-9 h-9 text-cyan/70 group-hover:text-cyan transition-colors" />
                </div>
                <p className="text-lg font-semibold text-white mb-2">{isDragActive ? 'Drop your PDF here' : 'Drag & drop a PDF file'}</p>
                <p className="text-sm text-gray-500">or <span className="text-accent-light underline underline-offset-2">browse files</span></p>
                <p className="text-xs text-gray-600 mt-4">Supports PDF files only</p>
              </div>
            )}

            {uploadState === 'uploading' && (
              <div className="animate-fade-in">
                <Loader2 className="w-12 h-12 text-amber-400 mx-auto mb-4 animate-spin" />
                <p className="text-lg font-semibold text-white mb-1">Ingesting Document...</p>
                <p className="text-sm text-gray-400 mb-4">{uploadInfo?.name}</p>
                <div className="w-full max-w-xs mx-auto h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-accent to-cyan rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
                </div>
                <p className="text-xs text-gray-500 mt-2">Extracting → Chunking → Embedding → Storing...</p>
              </div>
            )}

            {uploadState === 'success' && (
              <div className="animate-slide-up">
                <CheckCircle className="w-14 h-14 text-emerald-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-1">Successfully Ingested!</p>
                <p className="text-sm text-gray-400">{uploadInfo?.name}</p>
                <p className="text-xs text-emerald-400/80 mt-1">{uploadInfo?.message}</p>
                <button onClick={(e) => { e.stopPropagation(); reset() }} className="btn-primary mt-6 text-sm">Upload Another</button>
              </div>
            )}

            {uploadState === 'error' && (
              <div className="animate-slide-up">
                <XCircle className="w-14 h-14 text-red-400 mx-auto mb-4" />
                <p className="text-lg font-semibold text-white mb-1">Ingestion Failed</p>
                <p className="text-sm text-red-400/80">{uploadInfo?.error}</p>
                <button onClick={(e) => { e.stopPropagation(); reset() }} className="btn-primary mt-6 text-sm">Try Again</button>
              </div>
            )}
          </div>

          {/* Pipeline steps */}
          <div className="mt-8 glass-card">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">Ingestion Pipeline</h3>
            <div className="grid grid-cols-4 gap-4">
              {[
                { step: '1', label: 'Extract Text', desc: 'PyMuPDF' },
                { step: '2', label: 'Extract Tables', desc: 'Camelot' },
                { step: '3', label: 'Chunk', desc: 'Semantic split' },
                { step: '4', label: 'Embed & Store', desc: 'ChromaDB' },
              ].map((s) => (
                <div key={s.step} className="text-center">
                  <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center mx-auto mb-2 text-accent-light text-sm font-bold">{s.step}</div>
                  <p className="text-xs font-medium text-gray-300">{s.label}</p>
                  <p className="text-[10px] text-gray-600 mt-0.5">{s.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
