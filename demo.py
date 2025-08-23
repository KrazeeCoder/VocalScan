"""Demo script for VocalScan - showcases functionality without full ML dependencies."""

import json
import os
import sys
from datetime import datetime, timezone

# Simulate the VocalScan functionality
class VocalScanDemo:
    def __init__(self):
        self.version = "vocalscan-demo-v1.0"
    
    def analyze_audio(self, sample_type="voice", duration=10):
        """Simulate audio analysis."""
        
        # Simulate different results based on sample type
        if sample_type in ["cough", "breath"]:
            respiratory_score = 0.2 + (duration / 30) * 0.3  # Duration affects score
            neurological_score = 0.0
        elif sample_type in ["voice", "sustained", "sentence"]:
            respiratory_score = 0.0
            neurological_score = 0.15 + (duration / 20) * 0.2
        else:
            respiratory_score = 0.25
            neurological_score = 0.18
        
        # Ensure scores are in valid range
        respiratory_score = max(0.0, min(1.0, respiratory_score))
        neurological_score = max(0.0, min(1.0, neurological_score))
        
        scores = {
            "respiratory": round(respiratory_score, 3),
            "neurological": round(neurological_score, 3)
        }
        
        # Calculate risk level
        max_score = max(scores.values())
        if max_score < 0.33:
            risk_level = "low"
        elif max_score < 0.66:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Calculate confidence
        confidence = 0.75 + (duration / 30) * 0.2  # Longer samples = higher confidence
        confidence = max(0.5, min(0.95, confidence))
        
        # Generate interpretation
        interpretation = self._generate_interpretation(scores, risk_level, sample_type)
        
        return {
            "scores": scores,
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
            "model_version": self.version,
            "interpretation": interpretation,
            "sample_type": sample_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_interpretation(self, scores, risk_level, sample_type):
        """Generate interpretation of results."""
        
        interpretation = {
            "summary": "",
            "details": [],
            "nextSteps": [],
            "disclaimer": "This is a pattern analysis tool, not a medical diagnosis. Consult healthcare professionals for medical advice."
        }
        
        # Generate summary based on risk level
        if risk_level == "low":
            interpretation["summary"] = "Low likelihood of concerning patterns detected."
            interpretation["nextSteps"] = [
                "Continue regular health monitoring",
                "Consider periodic re-testing if symptoms develop"
            ]
        elif risk_level == "medium":
            interpretation["summary"] = "Some patterns of interest detected. Consider monitoring."
            interpretation["nextSteps"] = [
                "Monitor symptoms and voice changes",
                "Consider consultation with healthcare provider if patterns persist",
                "Retest in a few weeks"
            ]
        else:  # high
            interpretation["summary"] = "Notable patterns detected that may warrant attention."
            interpretation["nextSteps"] = [
                "Consider consultation with a healthcare provider",
                "Document any symptoms or voice changes",
                "Follow up testing may be beneficial"
            ]
        
        # Add specific details
        respiratory_score = scores.get("respiratory", 0)
        neurological_score = scores.get("neurological", 0)
        
        if sample_type in ["cough", "breath"] and respiratory_score > 0.3:
            interpretation["details"].append(
                f"Respiratory analysis shows patterns that may indicate breathing irregularities (score: {respiratory_score:.2f})"
            )
        
        if sample_type in ["voice", "sustained", "sentence"] and neurological_score > 0.3:
            interpretation["details"].append(
                f"Voice analysis shows patterns that may indicate vocal changes (score: {neurological_score:.2f})"
            )
        
        if not interpretation["details"]:
            interpretation["details"].append("Analysis shows patterns within normal ranges.")
        
        return interpretation

def demo_analysis():
    """Run a demo analysis."""
    print("🎤 VocalScan Demo Analysis")
    print("=" * 50)
    
    demo = VocalScanDemo()
    
    # Test different sample types
    sample_types = [
        ("voice", "General voice sample", 12),
        ("sustained", "Sustained 'aaah' sound", 15),
        ("cough", "Cough sample", 8),
        ("breath", "Breathing pattern", 20)
    ]
    
    for sample_type, description, duration in sample_types:
        print(f"\n📊 Analyzing: {description} ({duration}s)")
        print("-" * 30)
        
        result = demo.analyze_audio(sample_type, duration)
        
        print(f"Sample Type: {result['sample_type']}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Respiratory Score: {result['scores']['respiratory']:.3f}")
        print(f"Neurological Score: {result['scores']['neurological']:.3f}")
        print(f"Summary: {result['interpretation']['summary']}")
        
        if result['interpretation']['details']:
            print("Details:")
            for detail in result['interpretation']['details']:
                print(f"  • {detail}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"Model Version: {demo.version}")

def demo_api_response():
    """Show example API response format."""
    print("\n🔗 Example API Response Format")
    print("=" * 50)
    
    demo = VocalScanDemo()
    result = demo.analyze_audio("voice", 15)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    demo_analysis()
    demo_api_response()
    
    print("\n" + "=" * 50)
    print("🚀 VocalScan is ready!")
    print("\nTo start the full application:")
    print("1. Backend: start-backend.bat")
    print("2. Frontend: start-frontend.bat")
    print("3. Open: http://localhost:3000")
