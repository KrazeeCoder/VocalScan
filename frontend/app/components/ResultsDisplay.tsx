import { AlertCircle, CheckCircle, Clock, Activity, Brain, Lung } from 'lucide-react'

interface ResultsDisplayProps {
  results: any
  isLoading: boolean
}

export default function ResultsDisplay({ results, isLoading }: ResultsDisplayProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mb-4"></div>
        <p className="text-gray-600">Analyzing voice patterns...</p>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-500">
        <Activity size={48} className="mb-4 opacity-50" />
        <p className="text-center">
          Record or upload an audio sample to see analysis results
        </p>
      </div>
    )
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200'
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'high': return 'text-red-600 bg-red-50 border-red-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low': return <CheckCircle size={20} />
      case 'medium': return <Clock size={20} />
      case 'high': return <AlertCircle size={20} />
      default: return <Activity size={20} />
    }
  }

  const getScorePercentage = (score: number) => Math.round(score * 100)

  return (
    <div className="space-y-6">
      {/* Overall Risk Level */}
      <div className={`p-4 border-2 rounded-lg ${getRiskColor(results.riskLevel)}`}>
        <div className="flex items-center space-x-2 mb-2">
          {getRiskIcon(results.riskLevel)}
          <h3 className="font-semibold text-lg capitalize">
            {results.riskLevel} Risk Level
          </h3>
        </div>
        <p className="text-sm">
          {results.interpretation?.summary || 'Analysis complete'}
        </p>
      </div>

      {/* Detailed Scores */}
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-900">Pattern Analysis Scores</h4>
        
        {/* Respiratory Score */}
        {results.scores.respiratory > 0 && (
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Lung size={20} className="text-blue-600" />
                <span className="font-medium text-blue-900">Respiratory Patterns</span>
              </div>
              <span className="text-lg font-semibold text-blue-900">
                {getScorePercentage(results.scores.respiratory)}%
              </span>
            </div>
            <div className="w-full bg-blue-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${getScorePercentage(results.scores.respiratory)}%` }}
              ></div>
            </div>
            <p className="text-sm text-blue-700 mt-2">
              Analysis of breathing patterns and respiratory sounds
            </p>
          </div>
        )}

        {/* Neurological Score */}
        {results.scores.neurological > 0 && (
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Brain size={20} className="text-purple-600" />
                <span className="font-medium text-purple-900">Voice Quality Patterns</span>
              </div>
              <span className="text-lg font-semibold text-purple-900">
                {getScorePercentage(results.scores.neurological)}%
              </span>
            </div>
            <div className="w-full bg-purple-200 rounded-full h-2">
              <div 
                className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${getScorePercentage(results.scores.neurological)}%` }}
              ></div>
            </div>
            <p className="text-sm text-purple-700 mt-2">
              Analysis of voice quality and neurological voice markers
            </p>
          </div>
        )}
      </div>

      {/* Confidence Score */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium text-gray-700">Analysis Confidence</span>
          <span className="text-lg font-semibold text-gray-900">
            {getScorePercentage(results.confidence)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-gray-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${getScorePercentage(results.confidence)}%` }}
          ></div>
        </div>
      </div>

      {/* Interpretation */}
      {results.interpretation && (
        <div className="space-y-4">
          {results.interpretation.details && results.interpretation.details.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">Analysis Details</h4>
              <ul className="space-y-2">
                {results.interpretation.details.map((detail: string, index: number) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-sm text-gray-700">{detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {results.interpretation.nextSteps && results.interpretation.nextSteps.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">Recommended Next Steps</h4>
              <ul className="space-y-2">
                {results.interpretation.nextSteps.map((step: string, index: number) => (
                  <li key={index} className="flex items-start space-x-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{step}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Technical Details */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <details className="group">
          <summary className="cursor-pointer font-medium text-gray-700 hover:text-gray-900">
            Technical Details
          </summary>
          <div className="mt-2 space-y-2 text-sm text-gray-600">
            <div>Model Version: {results.modelVersion}</div>
            <div>Sample Type: {results.sampleType}</div>
            <div>Analysis Time: {new Date(results.timestamp).toLocaleString()}</div>
          </div>
        </details>
      </div>
    </div>
  )
}
