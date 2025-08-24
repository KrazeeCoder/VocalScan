(function(){
  async function authUser(){ return window.vsAuth.ensureAuthenticated(); }
  function toIsoStart(d){ return d ? new Date(`${d}T00:00:00`).toISOString() : ''; }
  function toIsoEnd(d){ return d ? new Date(`${d}T23:59:59`).toISOString() : ''; }
  async function getJSON(url){
    const user = await authUser();
    const token = await user.getIdToken();
    const res = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` }});
    const txt = await res.text();
    if (!res.ok) throw new Error(txt || 'Request failed');
    try { return JSON.parse(txt); } catch { return {}; }
  }
  function fmtLocal(iso, tz){
    if (!iso) return '';
    try{
      const dt = new Date(iso);
      return dt.toLocaleString(undefined, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' });
    } catch { return iso; }
  }
  function medHtml(d){
    const badge = d.taken ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700';
    const label = d.taken ? 'Taken' : 'Missed';
    const t = fmtLocal(d.timeRecordedIso || d.timeRecorded, d.timezone);
    return `<div class="p-4 border rounded flex items-center justify-between">
      <div>
        <div class="text-xs text-gray-500">${t} • ${d.timezone||''}</div>
        <div class="font-medium">${d.medName||d.medId||'Medication'}</div>
        <div class="text-sm text-gray-600">Sched: ${d.scheduledTime||'-'} ${d.notes?('• '+d.notes):''}</div>
      </div>
      <span class="px-2 py-0.5 rounded ${badge}">${label}</span>
    </div>`;
  }
  function symHtml(d){
    const t = fmtLocal(d.timeRecordedIso || d.timeRecorded, d.timezone);
    return `<div class="p-4 border rounded">
      <div class="text-xs text-gray-500">${t} • ${d.timezone||''}</div>
      <div class="flex flex-wrap gap-3 text-sm">
        <span class="px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">Symptoms</span>
        <span>Motor: ${d.motor??'-'}</span><span>Voice: ${d.voice??'-'}</span><span>Cog: ${d.cognition??'-'}</span><span>Mood: ${d.mood??'-'}</span><span>Fatigue: ${d.fatigue??'-'}</span><span>${d.onOff||''}</span>
      </div>
      <div class="text-sm text-gray-600">${d.notes||''}</div>
    </div>`;
  }
  async function load(){
    const user = await authUser();
    const from = document.getElementById('fromDate').value;
    const to = document.getElementById('toDate').value;
    const q = new URLSearchParams();
    const fromIso = toIsoStart(from); const toIso = toIsoEnd(to);
    if (fromIso) q.set('from', fromIso); if (toIso) q.set('to', toIso);
    const data = await getJSON(`/api/user/${user.uid}/timeline?${q.toString()}`);
    const meds = (data.medEvents||[]).map(e => ({ _type:'MED_EVENT', ...e }));
    const syms = (data.symptomLogs||[]).map(e => ({ _type:'SYMPTOM_LOG', ...e }));
    const items = meds.concat(syms).sort((a,b) => {
      const ta = Date.parse(a.timeRecordedIso || a.timeRecorded || 0);
      const tb = Date.parse(b.timeRecordedIso || b.timeRecorded || 0);
      return tb - ta;
    });
    const type = document.getElementById('typeFilter').value;
    const filtered = items.filter(i => type==='all' ? true : i._type===type);
    const container = document.getElementById('list');
    const empty = document.getElementById('emptyState');
    if (filtered.length === 0){ container.innerHTML = ''; empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden');
    container.innerHTML = filtered.map(d => d._type==='MED_EVENT'? medHtml(d) : symHtml(d)).join('');
  }
  window.addEventListener('DOMContentLoaded', async () => {
    await authUser();
    document.getElementById('applyBtn').onclick = () => load().catch(e => alert(e.message||'Failed'));
    load().catch(()=>{});
  });
})();


