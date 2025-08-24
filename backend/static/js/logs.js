(function(){
  async function authUser(){ return window.vsAuth.ensureAuthenticated(); }

  async function postJSON(url, body){
    const user = await authUser();
    const token = await user.getIdToken();
    const res = await fetch(url, { method:'POST', headers:{ 'Content-Type':'application/json', 'Authorization': `Bearer ${token}` }, body: JSON.stringify(body) });
    const txt = await res.text();
    if (!res.ok) throw new Error(txt || 'Request failed');
    try { return JSON.parse(txt); } catch { return { ok:true }; }
  }

  function tzName(){ try { return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'; } catch { return 'UTC'; } }
  function deviceId(){ return localStorage.getItem('vsDeviceId') || (()=>{ const id = 'dev_'+Math.random().toString(36).slice(2); localStorage.setItem('vsDeviceId', id); return id; })(); }

  async function saveMed(){
    const user = await authUser();
    const payload = {
      medId: document.getElementById('medId').value.trim() || undefined,
      scheduledTime: document.getElementById('scheduledTime').value.trim() || undefined,
      taken: document.getElementById('taken').checked,
      notes: document.getElementById('medNotes').value.trim() || undefined,
      timezone: tzName(),
      deviceId: deviceId(),
    };
    const json = await postJSON(`/api/user/${user.uid}/medEvent`, payload);
    document.getElementById('medMsg').textContent = `Saved: ${json.id}`;
  }

  async function saveSym(){
    const user = await authUser();
    const payload = {
      motor: Number(document.getElementById('motor').value),
      voice: Number(document.getElementById('voice').value),
      cognition: Number(document.getElementById('cognition').value),
      mood: Number(document.getElementById('mood').value),
      fatigue: Number(document.getElementById('fatigue').value),
      onOff: document.getElementById('onOff').value,
      notes: document.getElementById('symNotes').value.trim() || undefined,
      timezone: tzName(),
      deviceId: deviceId(),
    };
    const json = await postJSON(`/api/user/${user.uid}/symptomLog`, payload);
    document.getElementById('symMsg').textContent = `Saved: ${json.id}`;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await authUser();
    document.getElementById('saveMed').onclick = () => saveMed().catch(e => alert(e.message||'Failed'));
    document.getElementById('saveSym').onclick = () => saveSym().catch(e => alert(e.message||'Failed'));
    // Live value updates for sliders
    const ids = ['motor','voice','cognition','mood','fatigue'];
    ids.forEach(id => {
      const input = document.getElementById(id);
      const out = document.getElementById(id+'Val');
      if (input && out){
        const update = () => { out.textContent = input.value; };
        input.addEventListener('input', update);
        update();
      }
    });
  });
})();


