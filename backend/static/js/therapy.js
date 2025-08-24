(function(){
  const { auth, db, storage, storageRootRef } = window.vsFirebase;

  let mediaRecorder, chunks = [], startTime = 0, raf;
  let audioCtx, analyser, timeData;
  let baselineDb = -40; // simple UI baseline; could be personalized later

  const levelEl = document.getElementById('level');
  const dbLabel = document.getElementById('dbLabel');
  const timerEl = document.getElementById('timer');
  const statusEl = document.getElementById('therapyStatus');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const saveBtn = document.getElementById('saveBtn');
  const resetBtn = document.getElementById('resetBtn');
  const exerciseSel = document.getElementById('exercise');
  const targetDbInput = document.getElementById('targetDb');
  const ratingInput = document.getElementById('selfRating');
  const ratingVal = document.getElementById('ratingVal');
  const consentCk = document.getElementById('consent');

  ratingInput.oninput = () => ratingVal.textContent = ratingInput.value;

  function fmtTime(ms){ const s=Math.floor(ms/1000), m=Math.floor(s/60), r=s%60; return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`; }

  function setRecordingState(isRec){
    if (isRec) {
      startBtn.disabled = true; stopBtn.disabled = false; saveBtn.disabled = true;
      statusEl.classList.remove('hidden');
    } else {
      startBtn.disabled = false; stopBtn.disabled = true; saveBtn.disabled = chunks.length === 0;
      statusEl.classList.add('hidden');
    }
  }

  function computeDbFromTimeData(){
    let sum = 0; for (let i=0;i<timeData.length;i++){ const v = (timeData[i]-128)/128; sum += v*v; }
    const rms = Math.sqrt(sum / timeData.length);
    const db = 20*Math.log10(rms + 1e-8);
    return db;
  }

  function drawMeter(){
    raf = requestAnimationFrame(drawMeter);
    analyser.getByteTimeDomainData(timeData);
    const db = computeDbFromTimeData();
    dbLabel.textContent = `${db.toFixed(1)} dB`;
    const target = parseFloat(targetDbInput.value || '-20');
    const minDb = -60, maxDb = 0;
    const pct = Math.min(1, Math.max(0, (db - minDb) / (maxDb - minDb)));
    levelEl.style.width = (pct*100)+'%';
    // simple color hint by proximity to target
    const diff = Math.abs(db - target);
    levelEl.style.filter = diff < 3 ? 'saturate(1.4)' : diff < 8 ? 'saturate(1.0)' : 'grayscale(0.4)';
    timerEl.textContent = fmtTime(Date.now()-startTime);
  }

  async function start(){
    await window.vsAuth.ensureAuthenticatedWithDemographics();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    chunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    mediaRecorder.start(250);

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 2048; timeData = new Uint8Array(analyser.fftSize);
    source.connect(analyser);

    startTime = Date.now();
    setRecordingState(true);
    drawMeter();
  }

  function stop(){
    return new Promise(resolve => {
      if (!mediaRecorder) return resolve();
      mediaRecorder.onstop = () => { if (audioCtx) audioCtx.close(); cancelAnimationFrame(raf); setRecordingState(false); resolve(); };
      mediaRecorder.stop();
    });
  }

  async function save(){
    const user = auth.currentUser; if (!user) { location.href='/login'; return; }
    const exerciseType = exerciseSel.value;
    const durationSec = Math.max(0.1, (Date.now()-startTime)/1000);
    const selfRating = parseInt(ratingInput.value, 10) || 0;

    const sessionId = `s_${Date.now()}`;
    const doc = {
      exerciseType,
      timestamp: new Date().toISOString(),
      duration_seconds: Math.round(durationSec),
      target_db: parseFloat(targetDbInput.value || '-20'),
      selfRating,
      consentStored: !!consentCk.checked,
      processed: false
    };

    // Mirror Voice Recording flow: send file directly to backend to keep paths consistent
    const blob = new Blob(chunks, { type: 'audio/webm' });
    const token = await user.getIdToken();
    const fd = new FormData();
    const recordId = sessionId.replace('s_', 'rec_');
    fd.append('file', blob, `${recordId}.webm`);
    fd.append('sampleRate', String((audioCtx && audioCtx.sampleRate) || 48000));
    fd.append('durationSec', String(durationSec));
    fd.append('recordId', recordId);
    const resp = await fetch('/therapy/infer', { method: 'POST', headers: { 'Authorization': `Bearer ${token}` }, body: fd });
    const json = await resp.json();
    // Also persist minimal session doc for therapy page history
    doc.processed = true;
    doc.storagePath = json.storagePath || '';
    await db.collection('users').doc(user.uid).collection('voiceSessions').doc(sessionId).set(doc, { merge: true });
    showResult(json);
  }

  function reset(){
    chunks = []; saveBtn.disabled = true; document.getElementById('result').innerHTML = '<p class="text-dark-400">Ready.</p>';
  }

  function showResult(data){
    const res = document.getElementById('result');
    if (!data || data.error){
      res.innerHTML = `<p class="text-red-400">${data?.error || 'Processing failed'}</p>`;
      return;
    }
    const f = data.features || {};
    const s = typeof data.predicted_score === 'number' ? data.predicted_score : null;
    let level = 'No score';
    let levelClass = 'warning';
    let advisory = 'Record a few sessions to see a stable trend.';
    if (s !== null) {
      if (s < 0.33) { level = 'Good'; levelClass = 'success'; advisory = 'Low PD-likeness. Keep practicing and track over time.'; }
      else if (s < 0.66) { level = 'Medium'; levelClass = 'warning'; advisory = 'Mixed signal. Repeat under similar conditions and watch the weekly trend.'; }
      else { level = 'High'; levelClass = 'error'; advisory = 'Higher PD-likeness. If this persists across multiple sessions, share with a clinician.'; }
    }

    const bannerClass = levelClass === 'success' ? 'border-green-500/30 bg-green-500/10' : levelClass === 'warning' ? 'border-yellow-500/30 bg-yellow-500/10' : 'border-red-500/30 bg-red-500/10';

    res.innerHTML = `
      <div class="space-y-3">
        <div class="p-4 rounded-xl border ${bannerClass}">
          <div class="flex items-center justify-between mb-1">
            <span class="text-dark-300 text-sm">Overall</span>
            <span class="status-${levelClass}">${level}</span>
          </div>
          <div class="text-white font-bold text-xl">${s !== null ? `Score: ${s.toFixed(2)}` : 'Score unavailable'}</div>
          <p class="text-dark-300 text-sm mt-1">${advisory}</p>
        </div>

        <div class="p-3 bg-dark-800/50 rounded-xl"><span class="text-dark-300">RMS mean</span> <span class="float-right text-white">${(f.rms_db_mean??0).toFixed(2)} dB</span></div>
        <div class="p-3 bg-dark-800/50 rounded-xl"><span class="text-dark-300">F0 median</span> <span class="float-right text-white">${(f['MDVP:Fo(Hz)']??0).toFixed(1)} Hz</span></div>
        <div class="p-3 bg-dark-800/50 rounded-xl"><span class="text-dark-300">Jitter (%)</span> <span class="float-right text-white">${(f['MDVP:Jitter(%)']??0).toFixed(3)}</span></div>
        <div class="p-3 bg-dark-800/50 rounded-xl"><span class="text-dark-300">Shimmer (dB)</span> <span class="float-right text-white">${(f['MDVP:Shimmer(dB)']??0).toFixed(3)}</span></div>
      </div>
    `;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await window.vsAuth.ensureAuthenticatedWithDemographics();
    startBtn.onclick = start;
    stopBtn.onclick = async () => { await stop(); };
    saveBtn.onclick = save;
    resetBtn.onclick = reset;
    // enable Save once we have data
    const enableSave = () => { saveBtn.disabled = chunks.length === 0; };
    const origOnData = () => {};
    // hook to toggle save when chunks accumulate
    const observer = new MutationObserver(enableSave);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();


