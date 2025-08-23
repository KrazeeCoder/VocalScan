interface SampleTypeSelectorProps {
  value: string
  onChange: (value: string) => void
}

const sampleTypes = [
  { value: 'voice', label: 'Voice/Speech', description: 'General voice analysis' },
  { value: 'sustained', label: 'Sustained "Aaah"', description: 'Hold "aaah" sound for 10-15 seconds' },
  { value: 'sentence', label: 'Read Sentence', description: 'Read a provided sentence clearly' },
  { value: 'cough', label: 'Cough', description: 'Natural cough sounds' },
  { value: 'breath', label: 'Breathing', description: 'Deep breathing patterns' }
]

export default function SampleTypeSelector({ value, onChange }: SampleTypeSelectorProps) {
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-700 mb-3">
        Select Sample Type
      </label>
      <div className="grid grid-cols-1 gap-2">
        {sampleTypes.map((type) => (
          <label
            key={type.value}
            className={`flex items-center p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              value === type.value
                ? 'border-blue-500 bg-blue-50 text-blue-900'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <input
              type="radio"
              value={type.value}
              checked={value === type.value}
              onChange={(e) => onChange(e.target.value)}
              className="sr-only"
            />
            <div className="flex-1">
              <div className="font-medium">{type.label}</div>
              <div className="text-sm text-gray-500">{type.description}</div>
            </div>
            <div className={`w-4 h-4 rounded-full border-2 ${
              value === type.value
                ? 'border-blue-500 bg-blue-500'
                : 'border-gray-300'
            }`}>
              {value === type.value && (
                <div className="w-2 h-2 bg-white rounded-full m-auto mt-0.5"></div>
              )}
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}
