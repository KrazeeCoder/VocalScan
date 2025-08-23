import { useEffect, useRef } from 'react'

interface WaveformVisualizerProps {
  audioData: number[]
  isRecording: boolean
  duration: number
}

export default function WaveformVisualizer({ 
  audioData, 
  isRecording, 
  duration 
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * 2
    canvas.height = rect.height * 2
    ctx.scale(2, 2)

    // Clear canvas
    ctx.fillStyle = '#f8fafc'
    ctx.fillRect(0, 0, rect.width, rect.height)

    if (isRecording && audioData.length > 0) {
      // Draw real-time waveform
      const sliceWidth = rect.width / audioData.length
      const centerY = rect.height / 2

      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.beginPath()

      audioData.forEach((value, index) => {
        const x = index * sliceWidth
        const y = centerY + (value * centerY * 0.8)
        
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()
    } else if (!isRecording && duration > 0) {
      // Draw placeholder waveform for recorded audio
      const numBars = 50
      const barWidth = rect.width / numBars
      const centerY = rect.height / 2

      ctx.fillStyle = '#94a3b8'
      
      for (let i = 0; i < numBars; i++) {
        const height = Math.random() * (rect.height * 0.6) + 10
        const x = i * barWidth
        const y = centerY - height / 2
        
        ctx.fillRect(x, y, barWidth - 2, height)
      }
    } else {
      // Draw placeholder
      const centerY = rect.height / 2
      ctx.strokeStyle = '#cbd5e1'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(0, centerY)
      ctx.lineTo(rect.width, centerY)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#64748b'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Audio waveform will appear here', rect.width / 2, centerY - 10)
    }

  }, [audioData, isRecording, duration])

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="w-full h-32 border border-gray-200 rounded-lg waveform-canvas"
        style={{ height: '128px' }}
      />
      {isRecording && (
        <div className="absolute top-2 right-2 flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium text-red-600">REC</span>
        </div>
      )}
      {duration > 0 && (
        <div className="absolute bottom-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
          {duration}s
        </div>
      )}
    </div>
  )
}
