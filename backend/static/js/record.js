(function(){
  const { auth, storage } = window.vsFirebase;

  let mediaRecorder, chunks = [], startTime = 0, raf;
  let analyser, dataArray, canvas, ctx, audioCtx;

  function fmtTime(ms){
    const s = Math.floor(ms/1000); const m = Math.floor(s/60); const r = s%60;
    return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`;
  }

  async function ensureAuthed(){
    return new Promise(resolve => {
      auth.onAuthStateChanged(user => {
        if (!user) { location.href = '/login'; } else { resolve(user); }
      });
    });
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

    // Upload to Firebase Storage
    const storagePath = `audio/${user.uid}/rec_${recordId}.webm`;
    const ref = (window.vsFirebase.storageRootRef || storage.ref()).child(storagePath);
    await ref.put(blob, { contentType: 'audio/webm' });

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
    el.innerHTML = `
      <div class="card p-4">
        <div class="flex items-center justify-between">
          <div>
            <div class="text-sm text-gray-500">Risk</div>
            <div class="text-lg font-semibold">${result.riskLevel?.toUpperCase?.() || '-'}</div>
          </div>
          <div>
            <div class="text-sm text-gray-500">Confidence</div>
            <div class="text-lg font-semibold">${result.confidence ?? '-'}</div>
          </div>
          <div>
            <div class="text-sm text-gray-500">Resp</div>
            <div class="text-lg font-semibold">${result.scores?.respiratory ?? '-'}</div>
          </div>
          <div>
            <div class="text-sm text-gray-500">Neuro</div>
            <div class="text-lg font-semibold">${result.scores?.neurological ?? '-'}</div>
          </div>
        </div>
      </div>`;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await ensureAuthed();
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


