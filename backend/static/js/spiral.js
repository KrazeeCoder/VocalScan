(function(){
  const { auth, storage } = window.vsFirebase;

  let canvas, ctx, isDrawing = false, lastX = 0, lastY = 0;
  let hasStroke = false;
  let isCompleteAssessment = false;
  let voiceCompleted = false;

  // Check if this is part of a complete assessment flow
  function checkAssessmentFlow() {
    const urlParams = new URLSearchParams(window.location.search);
    isCompleteAssessment = urlParams.get('flow') === 'complete';
    voiceCompleted = urlParams.get('voice_completed') === 'true';
    
    if (isCompleteAssessment) {
      showAssessmentProgress();
    }
  }

  function showAssessmentProgress() {
    // Add progress bar at the top
    const progressHtml = `
      <div class="mb-6 p-4 bg-dark-800/50 border border-dark-600 rounded-xl">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-lg font-bold text-white">Complete Assessment Progress</h3>
          <span class="text-dark-300 font-medium">Step 2 of 2</span>
        </div>
        <div class="w-full bg-dark-700 rounded-full h-2">
          <div class="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full transition-all duration-500" style="width: 100%"></div>
        </div>
        <p class="text-dark-300 text-sm mt-2">
          <svg class="w-4 h-4 inline mr-1 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
          </svg>
          Voice Analysis Complete → Motor Function Test → Combined Results
        </p>
      </div>
    `;
    
    // Insert progress bar after the hero section
    const spiralContainer = document.querySelector('.max-w-4xl');
    if (spiralContainer) {
      spiralContainer.insertAdjacentHTML('afterbegin', progressHtml);
    }
  }

  function setupCanvas(){
    canvas = document.getElementById('spiralCanvas');
    ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#111827';
    drawGuides();

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      
      const x = (clientX - rect.left) * scaleX;
      const y = (clientY - rect.top) * scaleY;
      
      return { 
        x: Math.max(0, Math.min(canvas.width, x)), 
        y: Math.max(0, Math.min(canvas.height, y)) 
      };
    };

    const start = (e) => { isDrawing = true; hasStroke = true; const p = getPos(e); lastX = p.x; lastY = p.y; };
    const move = (e) => {
      if (!isDrawing) return;
      const p = getPos(e);
      ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(p.x, p.y); ctx.stroke();
      lastX = p.x; lastY = p.y; e.preventDefault();
    };
    const end = () => { isDrawing = false; };

    canvas.addEventListener('mousedown', start);
    canvas.addEventListener('mousemove', move);
    window.addEventListener('mouseup', end);
    canvas.addEventListener('touchstart', start, { passive:false });
    canvas.addEventListener('touchmove', move, { passive:false });
    window.addEventListener('touchend', end);
  }

  function drawGuides(){
    // No guides needed - using spiral.png template image instead
    // Canvas should remain transparent to show the spiral.png underneath
  }

  function clearCanvas(){
    ctx.clearRect(0, 0, canvas.width, canvas.height); drawGuides(); hasStroke = false;
  }

  async function uploadAndInfer(){
    const user = auth.currentUser; if (!user) { location.href='/login'; return; }
    if (!hasStroke) { alert('Please draw a spiral first.'); return; }
    
    const statusEl = document.getElementById('spiralStatus');
    const loadingEl = document.getElementById('spiralAnalysisLoading');
    
    // Show loading state
    if (loadingEl) loadingEl.classList.remove('hidden');
    
    statusEl.textContent = 'Uploading...';

    // Export canvas as PNG
    const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
    const drawingId = new Date().toISOString().replace(/[:.]/g,'-').replace('T','_').slice(0,19);
    const storagePath = `spirals/${user.uid}/spiral_${drawingId}.png`;
    const root = window.vsFirebase.storageRootRef || storage.ref();
    const ref = root.child(storagePath);
    await new Promise((resolve, reject) => {
      const task = ref.put(blob, { contentType: 'image/png' });
      task.on('state_changed',
        (snap) => {
          const pct = Math.round((snap.bytesTransferred / snap.totalBytes) * 100);
          statusEl.textContent = `Uploading ${pct}%...`;
        },
        (err) => { console.error('Spiral upload error:', err); statusEl.textContent = err.message || 'Upload failed'; reject(err); },
        () => resolve()
      );
    });

    // Call backend for placeholder analysis
    const token = await user.getIdToken();
    const fd = new FormData();
    fd.append('file', blob, 'spiral.png');
    fd.append('drawingId', `spiral_${drawingId}`);
    const res = await fetch('/spiral/infer', { method:'POST', headers:{ 'Authorization': `Bearer ${token}` }, body: fd, credentials: 'include' });
    if (!res.ok) { const t = await res.text(); throw new Error(t||'Analyze failed'); }
    const json = await res.json();
    showResult(json);
  }

  function showResult(result){
    // Hide loading state
    const loadingEl = document.getElementById('spiralAnalysisLoading');
    if (loadingEl) loadingEl.classList.add('hidden');
    
    // Clear status text
    const statusEl = document.getElementById('spiralStatus');
    if (statusEl) statusEl.textContent = '';
    
    // Reset submit button to normal state
    const submitBtn = document.getElementById('submitSpiralBtn');
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
        Save & Analyze
      `;
    }
    
    const el = document.getElementById('spiralResult');
    el.classList.remove('hidden');
    // Prefer PD probability if available from backend; fallback to confidence
    const pd = result?.scores?.pd;
    const pct = typeof pd === 'number' ? Math.round(pd * 1000) / 10 : (result.confidence ? Math.round(result.confidence * 1000) / 10 : null);
    el.innerHTML = `
      <div class="space-y-4">
        <div class="p-4 bg-dark-800/50 rounded-xl">
          <div class="flex items-center justify-between mb-2">
            <span class="text-dark-300 text-sm">Motor Control Score</span>
            <span class="status-${result.riskLevel === 'HIGH' ? 'error' : result.riskLevel === 'MEDIUM' ? 'warning' : 'success'}">
              ${result.riskLevel === 'HIGH' ? 'Needs Attention' : result.riskLevel === 'MEDIUM' ? 'Fair' : 'Good'}
            </span>
          </div>
          <div class="text-2xl font-bold text-white">${pct !== null ? pct.toFixed(1) + '%' : 'N/A'}</div>
          <div class="w-full bg-dark-700 rounded-full h-2 mt-2">
            <div class="bg-gradient-to-r from-green-500 to-primary-500 h-2 rounded-full transition-all duration-500" style="width: ${pct !== null ? pct : 75}%"></div>
          </div>
        </div>
        
        ${result.scores ? `
        <div class="grid grid-cols-1 gap-3">
          <div class="p-3 bg-dark-800/50 rounded-xl">
            <div class="flex justify-between items-center">
              <span class="text-dark-300 text-sm">Motor Control</span>
              <span class="text-white font-bold">${typeof result.scores.pd === 'number' ? (result.scores.pd * 100).toFixed(1) + '%' : (result.scores.respiratory ? (result.scores.respiratory * 100).toFixed(1) + '%' : 'N/A')}</span>
            </div>
          </div>
          <div class="p-3 bg-dark-800/50 rounded-xl">
            <div class="flex justify-between items-center">
              <span class="text-dark-300 text-sm">Tremor Detection</span>
              <span class="text-white font-bold">${result.scores.neurological ? (result.scores.neurological * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
          </div>
        </div>
        ` : ''}
        
        <div class="p-3 bg-primary-500/10 border border-primary-500/30 rounded-xl">
          <p class="text-primary-300 text-sm">
            <svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            Analysis complete. Results saved to your dashboard.
          </p>
        </div>
        
        ${isCompleteAssessment && voiceCompleted ? `
        <div class="mt-4 space-y-3">
          <div class="p-4 bg-green-500/10 border border-green-500/30 rounded-xl">
            <h4 class="text-green-400 font-bold mb-2">Complete Assessment Finished! 🎉</h4>
            <p class="text-green-300 text-sm">Both voice and motor function tests completed successfully.</p>
          </div>
          <button onclick="showCombinedResults()" class="w-full btn-primary px-6 py-3 text-center">
            <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            View Combined Assessment Results
          </button>
        </div>
        ` : ''}
      </div>`;
  }

  async function showCombinedResults() {
    try {
      // Show loading state
      const button = event.target;
      const originalText = button.innerHTML;
      button.innerHTML = `
        <svg class="w-5 h-5 inline mr-2 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
        </svg>
        Loading Combined Results...
      `;
      button.disabled = true;

      // Get recent voice and spiral results
      const user = auth.currentUser;
      const { db } = window.vsFirebase;
      
      if (!db) {
        throw new Error('Database not available');
      }
      
      // Fetch recent voice recording
      const voiceQuery = db.collection('users').doc(user.uid).collection('voiceRecordings')
        .orderBy('createdAt', 'desc').limit(1);
      const voiceSnapshot = await voiceQuery.get();
      
      // Fetch recent spiral drawing
      const spiralQuery = db.collection('users').doc(user.uid).collection('spiralDrawings')
        .orderBy('createdAt', 'desc').limit(1);
      const spiralSnapshot = await spiralQuery.get();
      
      let voiceResult = null;
      let spiralResult = null;
      
      if (!voiceSnapshot.empty) {
        voiceResult = voiceSnapshot.docs[0].data();
      }
      
      if (!spiralSnapshot.empty) {
        spiralResult = spiralSnapshot.docs[0].data();
      }
      
      // Calculate combined assessment
      const combinedAssessment = calculateCombinedRisk(voiceResult, spiralResult);
      
      // Display combined results
      displayCombinedResults(combinedAssessment, voiceResult, spiralResult);
      
    } catch (error) {
      console.error('Error fetching combined results:', error);
      
      // Restore button state
      const button = event.target;
      button.innerHTML = `
        <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
        </svg>
        View Combined Assessment Results
      `;
      button.disabled = false;
      
      // Show error message in the results area
      const resultContainer = document.getElementById('spiralResult');
      resultContainer.innerHTML += `
        <div class="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
          <p class="text-red-300 text-sm">
            <svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
            </svg>
            Unable to load combined results. Please try going to your dashboard to view all results.
          </p>
          <div class="mt-3">
            <a href="/dashboard" class="btn-secondary px-4 py-2 text-sm">Go to Dashboard</a>
          </div>
        </div>
      `;
    }
  }

  function calculateCombinedRisk(voice, spiral) {
    const voiceRisk = voice?.riskLevel || 'UNKNOWN';
    const spiralRisk = spiral?.riskLevel || 'UNKNOWN';
    
    const voiceConfidence = voice?.confidence || 0;
    const spiralConfidence = spiral?.confidence || 0;
    
    // Simple combination logic
    let combinedRiskLevel = 'LOW';
    let combinedConfidence = (voiceConfidence + spiralConfidence) / 2;
    
    if (voiceRisk === 'HIGH' || spiralRisk === 'HIGH') {
      combinedRiskLevel = 'HIGH';
    } else if (voiceRisk === 'MEDIUM' || spiralRisk === 'MEDIUM') {
      combinedRiskLevel = 'MEDIUM';
    }

    const recommendations = {
      'LOW': ['Continue regular monitoring', 'Maintain healthy lifestyle', 'Monthly check-ups recommended'],
      'MEDIUM': ['Schedule more frequent assessments', 'Consult with neurologist', 'Begin targeted exercises'],
      'HIGH': ['Seek immediate medical attention', 'Comprehensive neurological evaluation', 'Daily monitoring recommended']
    };

    return {
      riskLevel: combinedRiskLevel,
      confidence: combinedConfidence,
      recommendations: recommendations[combinedRiskLevel] || recommendations['LOW']
    };
  }

  function displayCombinedResults(combined, voice, spiral) {
    const resultContainer = document.getElementById('spiralResult');
    
    const riskColors = {
      'LOW': 'success',
      'MEDIUM': 'warning',
      'HIGH': 'error'
    };
    
    const riskColor = riskColors[combined.riskLevel] || 'success';
    
    resultContainer.innerHTML = `
      <div class="space-y-6">
        <!-- Combined Assessment Header -->
        <div class="text-center p-6 bg-gradient-to-br from-primary-500/10 to-purple-500/10 border border-primary-500/30 rounded-xl">
          <h3 class="text-2xl font-bold text-white mb-2">Complete Assessment Results</h3>
          <p class="text-dark-300">Combined analysis of voice and motor function tests</p>
        </div>

        <!-- Overall Risk Assessment -->
        <div class="p-6 bg-dark-800/50 border border-dark-600 rounded-xl">
          <h4 class="text-xl font-bold text-white mb-4 text-center">Overall Assessment</h4>
          <div class="grid grid-cols-2 gap-6 text-center">
            <div>
              <div class="text-3xl font-bold status-${riskColor} mb-2">${combined.riskLevel} RISK</div>
              <div class="text-dark-300 text-sm">Combined Risk Level</div>
            </div>
            <div>
              <div class="text-3xl font-bold text-primary-400 mb-2">${(combined.confidence * 100).toFixed(1)}%</div>
              <div class="text-dark-300 text-sm">Overall Confidence</div>
            </div>
          </div>
        </div>

        <!-- Individual Test Results -->
        <div class="grid md:grid-cols-2 gap-4">
          <!-- Voice Results -->
          <div class="p-4 bg-dark-800/50 border border-dark-600 rounded-xl">
            <h5 class="font-bold text-white mb-3 flex items-center">
              <svg class="w-5 h-5 mr-2 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
              </svg>
              Voice Analysis
            </h5>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-dark-300 text-sm">Risk Level:</span>
                <span class="status-${riskColors[voice?.riskLevel] || 'warning'}">${voice?.riskLevel || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-300 text-sm">Confidence:</span>
                <span class="text-white">${voice?.confidence ? (voice.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
              </div>
            </div>
          </div>

          <!-- Spiral Results -->
          <div class="p-4 bg-dark-800/50 border border-dark-600 rounded-xl">
            <h5 class="font-bold text-white mb-3 flex items-center">
              <svg class="w-5 h-5 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path>
              </svg>
              Motor Function
            </h5>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-dark-300 text-sm">Risk Level:</span>
                <span class="status-${riskColors[spiral?.riskLevel] || 'warning'}">${spiral?.riskLevel || 'Unknown'}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-300 text-sm">Confidence:</span>
                <span class="text-white">${spiral?.confidence ? (spiral.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Recommendations -->
        <div class="p-4 bg-dark-800/50 border border-dark-600 rounded-xl">
          <h5 class="font-bold text-white mb-3">Recommendations</h5>
          <ul class="space-y-2">
            ${combined.recommendations.map(rec => `
              <li class="flex items-start text-dark-300 text-sm">
                <svg class="w-4 h-4 mr-2 text-primary-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                ${rec}
              </li>
            `).join('')}
          </ul>
        </div>

        <!-- Action Buttons -->
        <div class="flex gap-3">
          <a href="/dashboard" class="flex-1 btn-primary px-6 py-3 text-center">
            <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            View Dashboard
          </a>
          <a href="/record?flow=complete" class="flex-1 btn-secondary px-6 py-3 text-center">
            <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            New Assessment
          </a>
        </div>
      </div>
    `;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await window.vsAuth.ensureAuthenticatedWithDemographics();
    checkAssessmentFlow();
    setupCanvas();
    document.getElementById('clearBtn').onclick = clearCanvas;
    document.getElementById('submitSpiralBtn').onclick = async () => {
      try {
        document.getElementById('submitSpiralBtn').disabled = true;
        await uploadAndInfer();
      } catch(e){ alert(e.message || 'Failed'); }
      finally { document.getElementById('submitSpiralBtn').disabled = false; }
    };
  });

  // Make showCombinedResults available globally
  window.showCombinedResults = showCombinedResults;
})();


