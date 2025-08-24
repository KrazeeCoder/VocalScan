(async function(){
  const { auth, db } = window.vsFirebase;

  function rowHtml(r){
    const t = r.createdAt?.toDate ? r.createdAt.toDate() : new Date();
    const s = r.scores || {}; const resp = s.respiratory ?? '-'; const neu = s.neurological ?? '-';
    return `<tr class="border-t"><td class="py-2">${t.toLocaleString?.() || '-'}</td><td>${r.riskLevel||'-'}</td><td>${r.confidence??'-'}</td><td>${resp}</td><td>${neu}</td></tr>`;
  }

  auth.onAuthStateChanged(async (user) => {
    if (!user) { location.href = '/login'; return; }
    const tbody = document.getElementById('historyRows');
    const colRef = db.collection('tests').doc(user.uid).collection('records').orderBy('createdAt','desc').limit(20);
    const snap = await colRef.get();
    tbody.innerHTML = snap.docs.map(d => rowHtml(d.data())).join('');

    const summary = document.getElementById('summary');
    const last = snap.docs[0]?.data();
    if (!last) { summary.textContent = 'No records yet.'; return; }
    summary.innerHTML = `Latest: <b>${last.riskLevel}</b> (conf ${last.confidence})<br/>Model: ${last.modelVersion}`;
  });
})();


