(function(){
  const { auth, storage } = window.vsFirebase;

  let canvas, ctx, isDrawing = false, lastX = 0, lastY = 0;
  let hasStroke = false;

  function ensureAuthed(){
    return new Promise(resolve => auth.onAuthStateChanged(u => { if(!u) location.href='/login'; else resolve(u); }));
  }

  function setupCanvas(){
    canvas = document.getElementById('spiralCanvas');
    ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#111827';
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    drawGuides();

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
      return { x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)) };
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
    // light spiral guideline
    ctx.save();
    ctx.globalAlpha = 0.15; ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 1.5;
    const cx = canvas.width/2, cy = canvas.height/2;
    ctx.beginPath();
    const turns = 6; const spacing = 8; let angle = 0; let radius = 2;
    const maxR = Math.min(cx, cy) - 10;
    while (radius < maxR){
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);
      if (angle === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      angle += Math.PI/36; radius += spacing/36;
    }
    ctx.stroke(); ctx.restore();
  }

  function clearCanvas(){
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height); drawGuides(); hasStroke = false;
  }

  async function uploadAndInfer(){
    const user = auth.currentUser; if (!user) { location.href='/login'; return; }
    if (!hasStroke) { alert('Please draw a spiral first.'); return; }
    const statusEl = document.getElementById('spiralStatus');
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
    statusEl.textContent = 'Saved & analyzed';
  }

  function showResult(result){
    const el = document.getElementById('spiralResult');
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
            <div class="text-sm text-gray-500">Score X</div>
            <div class="text-lg font-semibold">${result.scores?.respiratory ?? '-'}</div>
          </div>
          <div>
            <div class="text-sm text-gray-500">Score Y</div>
            <div class="text-lg font-semibold">${result.scores?.neurological ?? '-'}</div>
          </div>
        </div>
      </div>`;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await ensureAuthed();
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
})();


