(async function(){
  const { auth, db, storage } = window.vsFirebase;

  function rowHtml(r){
    const t = r.createdAt?.toDate ? r.createdAt.toDate() : new Date();
    const s = r.scores || {}; const resp = s.respiratory ?? '-'; const neu = s.neurological ?? '-';
    return `<tr class="border-t"><td class="py-2">${t.toLocaleString?.() || '-'}</td><td>${r.riskLevel||'-'}</td><td>${r.confidence??'-'}</td><td>${resp}</td><td>${neu}</td></tr>`;
  }

  async function loadVoice(user){
    const q = db.collection('users').doc(user.uid).collection('voiceRecordings').orderBy('createdAt','desc').limit(20);
    const snap = await q.get();
    return snap.docs.map(d => ({ id: d.id, ...d.data() }));
  }

  async function loadSpirals(user){
    const q = db.collection('users').doc(user.uid).collection('spiralDrawings').orderBy('createdAt','desc').limit(20);
    const snap = await q.get();
    return snap.docs.map(d => ({ id: d.id, ...d.data() }));
  }

  auth.onAuthStateChanged(async (user) => {
    if (!user) { location.href = '/login'; return; }
    const tbody = document.getElementById('historyRows');

    const [voice, spirals] = await Promise.all([loadVoice(user), loadSpirals(user)]);
    const all = [
      ...voice.map(v => ({ type:'voice', ...v })),
      ...spirals.map(s => ({ type:'spiral', ...s })),
    ].sort((a,b) => {
      const ta = a.createdAt?.toMillis ? a.createdAt.toMillis() : 0;
      const tb = b.createdAt?.toMillis ? b.createdAt.toMillis() : 0;
      return tb - ta;
    });

    tbody.innerHTML = all.map(r => rowHtml(r)).join('');

    const summary = document.getElementById('summary');
    if (all.length === 0) { summary.textContent = 'No submissions yet.'; return; }
    const last = all[0];
    summary.innerHTML = `Latest (${last.type}): <b>${last.riskLevel}</b> (conf ${last.confidence})<br/>Model: ${last.modelVersion}`;
  });
})();


