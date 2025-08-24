(function(){
  const { auth, storage } = window.vsFirebase;

  let mediaRecorder, chunks = [], startTime = 0, raf;
  let analyser, dataArray, canvas, ctx, audioCtx;
  let isCompleteAssessment = false;

  // Check if this is part of a complete assessment flow
  function checkAssessmentFlow() {
    const urlParams = new URLSearchParams(window.location.search);
    isCompleteAssessment = urlParams.get('flow') === 'complete';
    
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
          <span class="text-dark-300 font-medium">Step 1 of 2</span>
        </div>
        <div class="w-full bg-dark-700 rounded-full h-2">
          <div class="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full transition-all duration-500" style="width: 50%"></div>
        </div>
        <p class="text-dark-300 text-sm mt-2">Voice Analysis → Motor Function Test → Combined Results</p>
      </div>
    `;
    
    // Insert progress bar after the hero section
    const recordContainer = document.querySelector('.max-w-4xl');
    if (recordContainer) {
      recordContainer.insertAdjacentHTML('afterbegin', progressHtml);
    }
  }

  function fmtTime(ms){
    const s = Math.floor(ms/1000); const m = Math.floor(s/60); const r = s%60;
    return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`;
  }

  async function start(){
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = MediaRecorder.isTypeSupported('audio/webm; codecs=opus') ? 'audio/webm; codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    chunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    mediaRecorder.start(250);
    startVisualizer(stream);
    startTime = Date.now();
    toggleButtons(true);
  }

  function stop(){
    return new Promise(resolve => {
      if (!mediaRecorder) return resolve();
      mediaRecorder.onstop = () => { stopVisualizer(); toggleButtons(false); resolve(); };
      mediaRecorder.stop();
    });
  }

  function toggleButtons(recording){
    document.getElementById('startBtn').disabled = recording;
    document.getElementById('stopBtn').disabled = !recording;
    document.getElementById('submitBtn').disabled = recording || chunks.length === 0;
  }

  function startVisualizer(stream){
    canvas = document.getElementById('wave'); ctx = canvas.getContext('2d');
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    source.connect(analyser);
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const elapsed = Date.now()-startTime; document.getElementById('timer').textContent = fmtTime(elapsed);
      analyser.getByteTimeDomainData(dataArray);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.strokeStyle = '#4f46e5'; ctx.lineWidth = 2; ctx.beginPath();
      const sliceWidth = canvas.width / dataArray.length;
      for (let i=0;i<dataArray.length;i++){
        const x = i * sliceWidth; const v = dataArray[i] / 128.0; const y = v * canvas.height/2;
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }; draw();
  }

  function stopVisualizer(){ cancelAnimationFrame(raf); if (audioCtx) audioCtx.close(); }

  async function uploadAndInfer(){
    const user = auth.currentUser; if (!user) { location.href='/login'; return; }
    const durationSec = Math.max(0.1, (Date.now()-startTime)/1000);
    const sampleRate = Number(document.getElementById('sampleRate').value) || 48000;
    const recordId = new Date().toISOString().replace(/[:.]/g,'-').replace('T','_').slice(0,19);
    const blob = new Blob(chunks, { type: 'audio/webm' });

    // Upload to Firebase Storage with progress + correct bucket root
    const storagePath = `audio/${user.uid}/rec_${recordId}.webm`;
    const root = window.vsFirebase.storageRootRef || storage.ref();
    const ref = root.child(storagePath);
    await new Promise((resolve, reject) => {
      const task = ref.put(blob, { contentType: 'audio/webm' });
      task.on('state_changed',
        (snap) => {
          const pct = Math.round((snap.bytesTransferred / snap.totalBytes) * 100);
          document.getElementById('submitBtn').textContent = `Uploading ${pct}%...`;
        },
        (err) => { console.error('Audio upload error:', err); reject(err); },
        () => resolve()
      );
    });
    document.getElementById('submitBtn').textContent = 'Upload & Analyze';

    // Call backend /infer with ID token
    const token = await user.getIdToken();
    const fd = new FormData();
    fd.append('file', blob, 'recording.webm');
    fd.append('sampleRate', String(sampleRate));
    fd.append('durationSec', String(durationSec));
    fd.append('recordId', `rec_${recordId}`);

    const res = await fetch('/infer', { method: 'POST', headers: { 'Authorization': `Bearer ${token}` }, body: fd, credentials: 'include' });
    if (!res.ok) { const t = await res.text(); throw new Error(t || 'Inference failed'); }
    const json = await res.json();
    showResult(json);
  }

  function showResult(result){
    const el = document.getElementById('result');
    el.classList.remove('hidden');
    el.innerHTML = `
      <div class="space-y-4">
        <div class="p-4 bg-dark-800/50 rounded-xl">
          <div class="flex items-center justify-between mb-2">
            <span class="text-dark-300 text-sm">Risk Assessment</span>
            <span class="status-${result.riskLevel === 'HIGH' ? 'error' : result.riskLevel === 'MEDIUM' ? 'warning' : 'success'}">
              ${result.riskLevel || 'Unknown'}
            </span>
          </div>
          <div class="text-2xl font-bold text-white">${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
        </div>
        
        <div class="p-4 bg-dark-800/50 rounded-xl">
          <div class="flex items-center justify-between mb-2">
            <span class="text-dark-300 text-sm">Confidence Level</span>
          </div>
          <div class="text-2xl font-bold text-white">${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
          <div class="w-full bg-dark-700 rounded-full h-2 mt-2">
            <div class="bg-gradient-to-r from-primary-500 to-primary-600 h-2 rounded-full transition-all duration-500" style="width: ${result.confidence ? result.confidence * 100 : 0}%"></div>
          </div>
        </div>
        
        ${result.scores ? `
        <div class="grid grid-cols-2 gap-3">
          <div class="p-3 bg-dark-800/50 rounded-xl text-center">
            <div class="text-lg font-bold text-white">${result.scores.respiratory || 'N/A'}</div>
            <div class="text-xs text-dark-400">Respiratory</div>
          </div>
          <div class="p-3 bg-dark-800/50 rounded-xl text-center">
            <div class="text-lg font-bold text-white">${result.scores.neurological || 'N/A'}</div>
            <div class="text-xs text-dark-400">Neurological</div>
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
        
        ${isCompleteAssessment ? `
        <div class="mt-4">
          <a href="/spiral?flow=complete&voice_completed=true" class="w-full btn-primary px-6 py-3 text-center inline-flex items-center justify-center">
            Continue to Motor Function Test
            <svg class="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path>
            </svg>
          </a>
        </div>
        ` : ''}
      </div>`;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await window.vsAuth.ensureAuthenticatedWithDemographics();
    checkAssessmentFlow();
    document.getElementById('startBtn').onclick = start;
    document.getElementById('stopBtn').onclick = async () => { await stop(); document.getElementById('submitBtn').disabled = chunks.length === 0; };
    document.getElementById('submitBtn').onclick = async () => {
      try {
        document.getElementById('submitBtn').disabled = true;
        await uploadAndInfer();
      } catch (e) {
        alert(e.message || 'Failed');
      } finally {
        document.getElementById('submitBtn').disabled = false;
      }
    };
  });
})();


