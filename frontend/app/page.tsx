'use client'

import { useState, useRef, useEffect } from 'react'
import { Mic, MicOff, Play, Pause, RotateCcw, Upload } from 'lucide-react'
import WaveformVisualizer from './components/WaveformVisualizer'
import ResultsDisplay from './components/ResultsDisplay'
import SampleTypeSelector from './components/SampleTypeSelector'

export default function VocalScanApp() {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioData, setAudioData] = useState<number[]>([])
  const [results, setResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [sampleType, setSampleType] = useState('voice')
  const [recordingTime, setRecordingTime] = useState(0)
  const [error, setError] = useState('')

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  const startRecording = async () => {
    try {
      setError('')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      })
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      
      const audioContext = new AudioContext({ sampleRate: 16000 })
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)
      
      analyser.fftSize = 256
      source.connect(analyser)
      
      audioContextRef.current = audioContext
      analyserRef.current = analyser
      mediaRecorderRef.current = mediaRecorder
      
      const chunks: Blob[] = []
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data)
        }
      }
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        setAudioBlob(blob)
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      
      // Start timer
      intervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
      
      // Start visualizing
      visualizeAudio()
      
    } catch (err) {
      console.error('Error accessing microphone:', err)
      setError('Could not access microphone. Please check permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }

  const visualizeAudio = () => {
    if (!analyserRef.current) return
    
    const bufferLength = analyserRef.current.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    
    const draw = () => {
      if (!isRecording || !analyserRef.current) return
      
      analyserRef.current.getByteTimeDomainData(dataArray)
      const normalizedData = Array.from(dataArray).map(value => (value - 128) / 128)
      setAudioData(normalizedData)
      
      requestAnimationFrame(draw)
    }
    
    draw()
  }

  const playAudio = () => {
    if (!audioBlob) return
    
    const url = URL.createObjectURL(audioBlob)
    const audio = new Audio(url)
    audioElementRef.current = audio
    
    audio.onended = () => {
      setIsPlaying(false)
      URL.revokeObjectURL(url)
    }
    
    audio.play()
    setIsPlaying(true)
  }

  const pauseAudio = () => {
    if (audioElementRef.current) {
      audioElementRef.current.pause()
      setIsPlaying(false)
    }
  }

  const resetRecording = () => {
    setAudioBlob(null)
    setAudioData([])
    setResults(null)
    setRecordingTime(0)
    setError('')
    
    if (audioElementRef.current) {
      audioElementRef.current.pause()
      setIsPlaying(false)
    }
  }

  const analyzeAudio = async () => {
    if (!audioBlob) {
      setError('No audio to analyze')
      return
    }

    setIsAnalyzing(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.webm')
      formData.append('sampleRate', '16000')
      formData.append('durationSec', recordingTime.toString())
      formData.append('sampleType', sampleType)

      // For demo purposes, we'll use a mock token
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/predict`, {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer demo-token'
        },
        body: formData
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setResults(result)
    } catch (err) {
      console.error('Analysis error:', err)
      setError('Analysis failed. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const uploadFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      setAudioBlob(file)
      setRecordingTime(10) // Estimate
      setAudioData([]) // Clear visualization
      setError('')
    } else {
      setError('Please select a valid audio file')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            VocalScan
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered voice pattern analysis for health insights
          </p>
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Recording Section */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-2xl font-semibold mb-6 text-center">Record Your Sample</h2>
            
            {/* Sample Type Selector */}
            <SampleTypeSelector value={sampleType} onChange={setSampleType} />
            
            {/* Waveform Visualizer */}
            <div className="mb-6">
              <WaveformVisualizer 
                audioData={audioData} 
                isRecording={isRecording}
                duration={recordingTime}
              />
            </div>
            
            {/* Recording Controls */}
            <div className="flex justify-center space-x-4 mb-6">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="flex items-center space-x-2 bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-full transition-all duration-300 recording-button"
                >
                  <Mic size={20} />
                  <span>Start Recording</span>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex items-center space-x-2 bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-full transition-all duration-300 recording-button recording"
                >
                  <MicOff size={20} />
                  <span>Stop Recording ({recordingTime}s)</span>
                </button>
              )}
              
              {audioBlob && !isRecording && (
                <>
                  {!isPlaying ? (
                    <button
                      onClick={playAudio}
                      className="flex items-center space-x-2 bg-green-500 hover:bg-green-600 text-white px-4 py-3 rounded-full transition-all duration-300"
                    >
                      <Play size={20} />
                      <span>Play</span>
                    </button>
                  ) : (
                    <button
                      onClick={pauseAudio}
                      className="flex items-center space-x-2 bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-3 rounded-full transition-all duration-300"
                    >
                      <Pause size={20} />
                      <span>Pause</span>
                    </button>
                  )}
                  
                  <button
                    onClick={resetRecording}
                    className="flex items-center space-x-2 bg-gray-400 hover:bg-gray-500 text-white px-4 py-3 rounded-full transition-all duration-300"
                  >
                    <RotateCcw size={20} />
                    <span>Reset</span>
                  </button>
                </>
              )}
            </div>
            
            {/* File Upload Option */}
            <div className="border-t pt-4">
              <label className="flex items-center justify-center space-x-2 cursor-pointer bg-blue-50 hover:bg-blue-100 border-2 border-dashed border-blue-300 rounded-lg p-4 transition-all duration-300">
                <Upload size={20} className="text-blue-500" />
                <span className="text-blue-700">Or upload an audio file</span>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={uploadFile}
                  className="hidden"
                />
              </label>
            </div>
            
            {/* Analyze Button */}
            {audioBlob && !isRecording && (
              <div className="mt-6">
                <button
                  onClick={analyzeAudio}
                  disabled={isAnalyzing}
                  className="w-full bg-primary-500 hover:bg-primary-600 disabled:bg-gray-400 text-white py-3 px-6 rounded-full font-semibold transition-all duration-300"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Voice Pattern'}
                </button>
              </div>
            )}
            
            {/* Error Display */}
            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
          </div>
          
          {/* Results Section */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-2xl font-semibold mb-6 text-center">Analysis Results</h2>
            <ResultsDisplay results={results} isLoading={isAnalyzing} />
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800 text-center">
            <strong>Disclaimer:</strong> VocalScan is a pattern analysis tool for research and screening purposes only. 
            It does not provide medical diagnoses. Always consult healthcare professionals for medical advice.
          </p>
        </div>
      </div>
    </div>
  )
}
