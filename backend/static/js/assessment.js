// Complete Assessment Flow - Guided voice and motor assessment
(async function(){
  const { auth, db, storage } = window.vsFirebase;

  let assessmentData = {
    voice: null,
    spiral: null,
    userId: null,
    startTime: new Date()
  };

  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  let recordingStartTime;

  // Canvas variables for spiral drawing
  let canvas, ctx;
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let spiralPath = [];

  // Initialize assessment flow
  auth.onAuthStateChanged(async (user) => {
    if (!user) {
      location.href = '/login';
      return;
    }

    assessmentData.userId = user.uid;
    
    // Check demographics completion
    try {
      const idToken = await user.getIdToken();
      const response = await fetch('/demographics/status', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${idToken}`
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        if (!result.demographicsCompleted) {
          location.href = '/demographics';
          return;
        }
      } else {
        location.href = '/demographics';
        return;
      }
    } catch (error) {
      console.error('Error checking demographics status:', error);
      location.href = '/demographics';
      return;
    }

    initializeAssessment();
  });

  function initializeAssessment() {
    console.log('Initializing complete assessment flow');
    
    // Initialize voice recording
    initializeVoiceRecording();
    
    // Initialize spiral drawing
    initializeSpiralDrawing();
    
    // Set up step navigation
    setupStepNavigation();
  }

  // === VOICE RECORDING FUNCTIONALITY ===
  function initializeVoiceRecording() {
    const startBtn = document.getElementById('voiceStartBtn');
    const stopBtn = document.getElementById('voiceStopBtn');
    const nextBtn = document.getElementById('voiceNextBtn');

    startBtn.addEventListener('click', startVoiceRecording);
    stopBtn.addEventListener('click', stopVoiceRecording);
    nextBtn.addEventListener('click', () => goToStep(2));
  }

  async function startVoiceRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await processVoiceRecording(audioBlob);
      };

      mediaRecorder.start();
      isRecording = true;
      recordingStartTime = Date.now();

      // Update UI
      document.getElementById('voiceStartBtn').classList.add('hidden');
      document.getElementById('voiceStopBtn').classList.remove('hidden');
      document.getElementById('voiceStatus').classList.remove('hidden');

      // Start progress animation
      animateRecordingProgress();

    } catch (error) {
      console.error('Error starting voice recording:', error);
      showError('Could not access microphone. Please check permissions.');
    }
  }

  function stopVoiceRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      
      // Stop all tracks
      mediaRecorder.stream.getTracks().forEach(track => track.stop());

      // Update UI
      document.getElementById('voiceStopBtn').classList.add('hidden');
      document.getElementById('voiceStatus').classList.add('hidden');
    }
  }

  function animateRecordingProgress() {
    if (!isRecording) return;

    const elapsed = Date.now() - recordingStartTime;
    const maxDuration = 30000; // 30 seconds max
    const progress = Math.min((elapsed / maxDuration) * 100, 100);

    document.getElementById('voiceProgress').style.width = progress + '%';

    if (progress >= 100) {
      stopVoiceRecording();
      return;
    }

    if (isRecording) {
      requestAnimationFrame(animateRecordingProgress);
    }
  }

  async function processVoiceRecording(audioBlob) {
    try {
      // Upload audio to server for processing
      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice_recording.webm');

      const user = auth.currentUser;
      const idToken = await user.getIdToken();

      const response = await fetch('/infer', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${idToken}`
        },
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        assessmentData.voice = result;
        
        // Show completion
        document.getElementById('voiceComplete').classList.remove('hidden');
        
        console.log('Voice analysis result:', result);
      } else {
        throw new Error('Voice analysis failed');
      }
    } catch (error) {
      console.error('Error processing voice recording:', error);
      showError('Voice analysis failed. Please try again.');
    }
  }

  // === SPIRAL DRAWING FUNCTIONALITY ===
  function initializeSpiralDrawing() {
    canvas = document.getElementById('spiralCanvas');
    ctx = canvas.getContext('2d');

    // Set up canvas
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#1e293b';

    // Draw guide spiral
    drawGuideSpiral();

    // Event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    // Buttons
    document.getElementById('spiralClearBtn').addEventListener('click', clearSpiral);
    document.getElementById('spiralSubmitBtn').addEventListener('click', submitSpiral);
    document.getElementById('spiralNextBtn').addEventListener('click', () => goToStep(3));
  }

  function drawGuideSpiral() {
    ctx.save();
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const maxRadius = 200;
    const turns = 3;

    ctx.beginPath();
    for (let angle = 0; angle <= turns * 2 * Math.PI; angle += 0.1) {
      const radius = (angle / (turns * 2 * Math.PI)) * maxRadius;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      if (angle === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = e.clientX - rect.left;
    lastY = e.clientY - rect.top;
    
    spiralPath.push({ x: lastX, y: lastY, timestamp: Date.now() });
    
    // Enable submit button after first stroke
    document.getElementById('spiralSubmitBtn').disabled = false;
    document.getElementById('spiralSubmitBtn').classList.remove('opacity-50', 'cursor-not-allowed');
  }

  function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    spiralPath.push({ x: currentX, y: currentY, timestamp: Date.now() });

    lastX = currentX;
    lastY = currentY;
  }

  function stopDrawing() {
    isDrawing = false;
  }

  function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  }

  function clearSpiral() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGuideSpiral();
    spiralPath = [];
    
    // Disable submit button
    document.getElementById('spiralSubmitBtn').disabled = true;
    document.getElementById('spiralSubmitBtn').classList.add('opacity-50', 'cursor-not-allowed');
  }

  async function submitSpiral() {
    if (spiralPath.length === 0) {
      showError('Please draw a spiral before submitting.');
      return;
    }

    try {
      // Convert canvas to image
      const imageDataUrl = canvas.toDataURL('image/png');
      
      // Prepare spiral data
      const spiralData = {
        path: spiralPath,
        image: imageDataUrl,
        timestamp: new Date().toISOString()
      };

      const user = auth.currentUser;
      const idToken = await user.getIdToken();

      // Submit to server
      const response = await fetch('/api/analyze-spiral', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`
        },
        body: JSON.stringify(spiralData)
      });

      if (response.ok) {
        const result = await response.json();
        assessmentData.spiral = result;
        
        // Show completion
        document.getElementById('spiralComplete').classList.remove('hidden');
        
        console.log('Spiral analysis result:', result);
      } else {
        throw new Error('Spiral analysis failed');
      }
    } catch (error) {
      console.error('Error submitting spiral:', error);
      showError('Spiral analysis failed. Please try again.');
    }
  }

  // === STEP NAVIGATION ===
  function setupStepNavigation() {
    // New assessment button
    document.getElementById('newAssessmentBtn').addEventListener('click', () => {
      if (confirm('Start a new assessment? This will reset your current progress.')) {
        location.reload();
      }
    });
  }

  function goToStep(stepNumber) {
    // Hide all steps
    document.querySelectorAll('.assessment-step').forEach(step => {
      step.classList.add('hidden');
    });

    // Show target step
    document.getElementById(`step${stepNumber}`).classList.remove('hidden');

    // Update progress
    updateProgress(stepNumber);

    // Handle step-specific logic
    if (stepNumber === 3) {
      generateCombinedResults();
    }
  }

  function updateProgress(step) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    const percentage = (step / 3) * 100;
    progressBar.style.width = percentage + '%';
    
    const stepTexts = {
      1: 'Step 1 of 3 - Voice Analysis',
      2: 'Step 2 of 3 - Motor Test',
      3: 'Step 3 of 3 - Results'
    };
    
    progressText.textContent = stepTexts[step];
  }

  // === COMBINED RESULTS ===
  async function generateCombinedResults() {
    try {
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));

      const voiceResult = assessmentData.voice;
      const spiralResult = assessmentData.spiral;

      // Calculate combined risk assessment
      const combinedRisk = calculateCombinedRisk(voiceResult, spiralResult);
      
      // Display results
      displayCombinedResults(combinedRisk, voiceResult, spiralResult);

      // Save to Firebase
      await saveCombinedAssessment(combinedRisk);

    } catch (error) {
      console.error('Error generating combined results:', error);
      showError('Error generating results. Please try again.');
    }
  }

  function calculateCombinedRisk(voice, spiral) {
    // Combine voice and spiral results for overall assessment
    const voiceRisk = voice?.riskLevel || 'UNKNOWN';
    const spiralRisk = spiral?.riskLevel || 'UNKNOWN';
    
    const voiceConfidence = voice?.confidence || 0;
    const spiralConfidence = spiral?.confidence || 0;
    
    // Simple combination logic - can be made more sophisticated
    let combinedRiskLevel = 'LOW';
    let combinedConfidence = (voiceConfidence + spiralConfidence) / 2;
    
    if (voiceRisk === 'HIGH' || spiralRisk === 'HIGH') {
      combinedRiskLevel = 'HIGH';
    } else if (voiceRisk === 'MEDIUM' || spiralRisk === 'MEDIUM') {
      combinedRiskLevel = 'MEDIUM';
    }

    return {
      riskLevel: combinedRiskLevel,
      confidence: combinedConfidence,
      voiceAnalysis: voice,
      spiralAnalysis: spiral,
      recommendations: generateRecommendations(combinedRiskLevel)
    };
  }

  function generateRecommendations(riskLevel) {
    const recommendations = {
      'LOW': [
        'Continue regular monitoring with monthly assessments',
        'Maintain a healthy lifestyle with regular exercise',
        'Consider voice exercises to maintain vocal strength'
      ],
      'MEDIUM': [
        'Schedule more frequent assessments (bi-weekly)',
        'Consult with a neurologist for professional evaluation',
        'Begin targeted exercises for motor and voice function'
      ],
      'HIGH': [
        'Seek immediate medical attention from a specialist',
        'Schedule comprehensive neurological evaluation',
        'Begin monitoring tremor and movement patterns daily'
      ]
    };
    
    return recommendations[riskLevel] || recommendations['LOW'];
  }

  function displayCombinedResults(combinedRisk, voiceResult, spiralResult) {
    document.getElementById('resultsLoading').classList.add('hidden');
    
    const resultsContainer = document.getElementById('combinedResults');
    
    const riskColors = {
      'LOW': 'success',
      'MEDIUM': 'warning',
      'HIGH': 'error'
    };
    
    const riskColor = riskColors[combinedRisk.riskLevel] || 'success';
    
    resultsContainer.innerHTML = `
      <!-- Overall Risk Assessment -->
      <div class="bg-dark-800/50 border border-dark-600 rounded-xl p-6 mb-6">
        <h3 class="text-xl font-bold text-white mb-4">Overall Assessment</h3>
        <div class="grid md:grid-cols-2 gap-6">
          <div class="text-center">
            <div class="text-3xl font-bold status-${riskColor} mb-2">${combinedRisk.riskLevel} RISK</div>
            <div class="text-dark-300 text-sm">Combined Risk Level</div>
          </div>
          <div class="text-center">
            <div class="text-3xl font-bold text-primary-400 mb-2">${(combinedRisk.confidence * 100).toFixed(1)}%</div>
            <div class="text-dark-300 text-sm">Overall Confidence</div>
          </div>
        </div>
      </div>

      <!-- Individual Test Results -->
      <div class="grid md:grid-cols-2 gap-6 mb-6">
        <!-- Voice Results -->
        <div class="bg-dark-800/50 border border-dark-600 rounded-xl p-6">
          <h4 class="text-lg font-bold text-white mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
            </svg>
            Voice Analysis
          </h4>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-dark-300">Risk Level:</span>
              <span class="status-${riskColors[voiceResult?.riskLevel] || 'warning'}">${voiceResult?.riskLevel || 'Unknown'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-dark-300">Confidence:</span>
              <span class="text-white">${voiceResult?.confidence ? (voiceResult.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>

        <!-- Spiral Results -->
        <div class="bg-dark-800/50 border border-dark-600 rounded-xl p-6">
          <h4 class="text-lg font-bold text-white mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path>
            </svg>
            Motor Function
          </h4>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-dark-300">Risk Level:</span>
              <span class="status-${riskColors[spiralResult?.riskLevel] || 'warning'}">${spiralResult?.riskLevel || 'Unknown'}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-dark-300">Confidence:</span>
              <span class="text-white">${spiralResult?.confidence ? (spiralResult.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Recommendations -->
      <div class="bg-dark-800/50 border border-dark-600 rounded-xl p-6">
        <h4 class="text-lg font-bold text-white mb-4">Recommendations</h4>
        <ul class="space-y-2">
          ${combinedRisk.recommendations.map(rec => `
            <li class="flex items-start text-dark-300">
              <svg class="w-5 h-5 mr-2 text-primary-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
              </svg>
              ${rec}
            </li>
          `).join('')}
        </ul>
      </div>
    `;
  }

  async function saveCombinedAssessment(combinedRisk) {
    try {
      const user = auth.currentUser;
      
      // Save to Firestore
      await db.collection('users').doc(user.uid).collection('assessments').add({
        type: 'complete',
        voiceData: assessmentData.voice,
        spiralData: assessmentData.spiral,
        combinedResults: combinedRisk,
        createdAt: new Date(),
        startTime: assessmentData.startTime,
        completedAt: new Date()
      });

      console.log('Combined assessment saved successfully');
    } catch (error) {
      console.error('Error saving assessment:', error);
    }
  }

  function showError(message) {
    // Simple error display - could be enhanced with toast notifications
    alert(message);
  }

})();
