(function(){
  async function authUser(){ return window.vsAuth.ensureAuthenticated(); }
  function buildQuery(from, to, extra){
    const q = new URLSearchParams();
    if (from) q.set('from', from);
    if (to) q.set('to', to);
    for (const [k,v] of Object.entries(extra||{})) if (v!=null && v!=='') q.set(k, String(v));
    return q.toString();
  }
  async function downloadOrLink(url){
    const user = await authUser();
    const token = await user.getIdToken();
    const res = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` }});
    const contentType = res.headers.get('content-type')||'';
    if (contentType.includes('application/json')){
      const data = await res.json();
      return { link: data.url, path: data.path };
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = url.includes('pdf') ? 'report.pdf' : 'export.csv';
    document.body.appendChild(a); a.click(); a.remove();
    return {};
  }

  async function csv(){
    const user = await authUser();
    const from = document.getElementById('csvFrom').value.trim();
    const to = document.getElementById('csvTo').value.trim();
    const save = document.getElementById('csvSave').checked;
    const q = buildQuery(from, to, { save });
    const url = `/api/user/${user.uid}/export/csv?${q}`;
    const res = await downloadOrLink(url);
    document.getElementById('csvMsg').innerHTML = res.link ? `<a class="text-indigo-700 underline" target="_blank" href="${res.link}">Download link</a>` : 'Downloaded';
  }

  async function pdf(){
    const user = await authUser();
    const from = document.getElementById('pdfFrom').value.trim();
    const to = document.getElementById('pdfTo').value.trim();
    const includeAudio = document.getElementById('pdfAudio').checked;
    const save = document.getElementById('pdfSave').checked;
    const q = buildQuery(from, to, { includeAudio, save });
    const url = `/api/user/${user.uid}/export/pdf?${q}`;
    const res = await downloadOrLink(url);
    document.getElementById('pdfMsg').innerHTML = res.link ? `<a class="text-indigo-700 underline" target="_blank" href="${res.link}">Download link</a>` : 'Downloaded';
  }

  async function share(){
    const user = await authUser();
    const storagePath = document.getElementById('sharePath').value.trim();
    const expiresMinutes = document.getElementById('shareExp').value.trim();
    const token = await user.getIdToken();
    const res = await fetch(`/api/user/${user.uid}/share`, { method:'POST', headers:{ 'Content-Type':'application/json', 'Authorization': `Bearer ${token}` }, body: JSON.stringify({ storagePath, expiresMinutes: expiresMinutes || undefined }) });
    if (!res.ok){ const t = await res.text(); throw new Error(t||'Share failed'); }
    const json = await res.json();
    document.getElementById('shareMsg').innerHTML = `<a class="text-indigo-700 underline" target="_blank" href="${json.url}">Share link</a> (expires in ${json.expiresMinutes}m)`;
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await authUser();
    document.getElementById('csvBtn').onclick = () => csv().catch(e => alert(e.message||'Failed'));
    document.getElementById('pdfBtn').onclick = () => pdf().catch(e => alert(e.message||'Failed'));
    document.getElementById('shareBtn').onclick = () => share().catch(e => alert(e.message||'Failed'));
  });
})();


