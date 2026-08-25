/**
 * ImageUploader
 * ─────────────
 * Drag-and-drop zone built on react-dropzone.
 * Accepts PNG / JPEG images up to 5 MB.
 * Shows a preview thumbnail after selection.
 *
 * Props:
 *   onFileSelect(file: File) — called when the user selects an image
 *   disabled?: boolean
 */
import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X } from 'lucide-react'

export default function ImageUploader({ onFileSelect, disabled = false }) {
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    setPreview(URL.createObjectURL(file))
    setFileName(file.name)
    onFileSelect(file)
  }, [onFileSelect])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/png': [], 'image/jpeg': [], 'image/jpg': [] },
    maxSize: 5 * 1024 * 1024,
    multiple: false,
    disabled,
  })

  const clear = (e) => {
    e.stopPropagation()
    setPreview(null)
    setFileName(null)
    onFileSelect(null)
  }

  return (
    <div
      {...getRootProps()}
      className={`
        relative flex flex-col items-center justify-center gap-3
        rounded-xl border-2 border-dashed p-8 cursor-pointer
        transition-all duration-200
        ${isDragActive
          ? 'border-brand-500 bg-brand-500/10'
          : 'border-gray-700 hover:border-brand-500/60 hover:bg-white/[0.02]'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input {...getInputProps()} />

      {preview ? (
        <>
          <img src={preview} alt="Preview" className="h-40 w-40 object-contain rounded-lg" />
          <p className="text-sm text-gray-400 font-mono truncate max-w-xs">{fileName}</p>
          <button
            onClick={clear}
            className="absolute top-3 right-3 rounded-full bg-gray-800 p-1 hover:bg-red-500/80 transition-colors"
          >
            <X size={14} />
          </button>
        </>
      ) : (
        <>
          {isDragActive ? (
            <ImageIcon size={40} className="text-brand-500" />
          ) : (
            <Upload size={40} className="text-gray-500" />
          )}
          <p className="text-sm text-gray-400 text-center">
            {isDragActive
              ? 'Drop the image here'
              : 'Drag & drop a digit image, or click to browse'}
          </p>
          <p className="text-xs text-gray-600">PNG / JPEG · max 5 MB</p>
        </>
      )}
    </div>
  )
}
