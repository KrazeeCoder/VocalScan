(function(){
  async function authUser(){ return window.vsAuth.ensureAuthenticated(); }
  function buildQuery(fromDate, toDate, extra){
    const q = new URLSearchParams();
    // Convert YYYY-MM-DD to ISO range in local timezone for backend
    function toIsoStart(d){ return d ? new Date(`${d}T00:00:00`).toISOString() : ''; }
    function toIsoEnd(d){ return d ? new Date(`${d}T23:59:59`).toISOString() : ''; }
    const from = toIsoStart(fromDate);
    const to = toIsoEnd(toDate);
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
    const from = document.getElementById('csvFrom').value;
    const to = document.getElementById('csvTo').value;
    const save = document.getElementById('csvSave').checked;
    const q = buildQuery(from, to, { save });
    const url = `/api/user/${user.uid}/export/csv?${q}`;
    const res = await downloadOrLink(url);
    document.getElementById('csvMsg').innerHTML = res.link ? `<a class="text-indigo-700 underline" target="_blank" href="${res.link}">Download link</a>` : 'Downloaded';
  }

  async function pdf(){
    const user = await authUser();
    const from = document.getElementById('pdfFrom').value;
    const to = document.getElementById('pdfTo').value;
    const includeAudio = document.getElementById('pdfAudio').checked;
    const save = document.getElementById('pdfSave').checked;
    const q = buildQuery(from, to, { includeAudio, save });
    const url = `/api/user/${user.uid}/export/pdf?${q}`;
    const res = await downloadOrLink(url);
    document.getElementById('pdfMsg').innerHTML = res.link ? `<a class="text-indigo-700 underline" target="_blank" href="${res.link}">Download link</a>` : 'Downloaded';
  }

  window.addEventListener('DOMContentLoaded', async () => {
    await authUser();
    document.getElementById('csvBtn').onclick = () => csv().catch(e => alert(e.message||'Failed'));
    document.getElementById('pdfBtn').onclick = () => pdf().catch(e => alert(e.message||'Failed'));
  });
})();


