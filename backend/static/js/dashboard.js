(async function(){
  const { auth, db, storage } = window.vsFirebase;

  function rowHtml(r){
    const t = r.createdAt?.toDate ? r.createdAt.toDate() : new Date();
    const s = r.scores || {}; 
    const resp = s.respiratory ?? '-'; 
    const neu = s.neurological ?? '-';
    const riskClass = r.riskLevel === 'HIGH' ? 'error' : r.riskLevel === 'MEDIUM' ? 'warning' : 'success';
    
    return `
      <tr class="hover:bg-dark-800/30 transition-colors">
        <td class="py-4 px-4 text-dark-200">${t.toLocaleString?.() || '-'}</td>
        <td class="py-4 px-4">
          <span class="px-2 py-1 text-xs rounded-full bg-${r.type === 'voice' ? 'primary' : 'purple'}-500/20 text-${r.type === 'voice' ? 'primary' : 'purple'}-300 border border-${r.type === 'voice' ? 'primary' : 'purple'}-500/30">
            ${r.type === 'voice' ? 'Voice' : 'Motor'}
          </span>
        </td>
        <td class="py-4 px-4">
          <span class="status-${riskClass}">${r.riskLevel || '-'}</span>
        </td>
        <td class="py-4 px-4 text-dark-200">${r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '-'}</td>
        <td class="py-4 px-4">
          <span class="status-${resp === '-' ? 'warning' : 'success'}">${resp === '-' ? 'Pending' : 'Complete'}</span>
        </td>
        <td class="py-4 px-4">
          <button class="text-primary-400 hover:text-primary-300 text-sm font-medium">
            View Details
          </button>
        </td>
      </tr>
    `;
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

  function updateMetrics(assessments) {
    const totalEl = document.getElementById('totalAssessments');
    const latestRiskEl = document.getElementById('latestRisk');
    const avgConfidenceEl = document.getElementById('avgConfidence');
    const lastAssessmentEl = document.getElementById('lastAssessment');

    if (totalEl) totalEl.textContent = assessments.length;

    if (assessments.length > 0) {
      const latest = assessments[0];
      if (latestRiskEl) latestRiskEl.textContent = latest.riskLevel || '--';
      
      const avgConf = assessments.reduce((sum, a) => sum + (a.confidence || 0), 0) / assessments.length;
      if (avgConfidenceEl) avgConfidenceEl.textContent = (avgConf * 100).toFixed(1) + '%';
      
      const lastDate = latest.createdAt?.toDate ? latest.createdAt.toDate() : new Date();
      if (lastAssessmentEl) lastAssessmentEl.textContent = lastDate.toLocaleDateString();
    }
  }

  function updateSummary(assessments) {
    const summary = document.getElementById('summaryDetails');
    const healthBar = document.getElementById('healthBar');
    const overallStatus = document.getElementById('overallStatus');
    
    if (assessments.length === 0) { 
      if (summary) summary.innerHTML = '<p class="text-dark-400 text-sm">Complete your first assessment to see health insights.</p>';
      return; 
    }
    
    const latest = assessments[0];
    const riskLevel = latest.riskLevel;
    
    // Update health bar and status
    let healthPercentage = 75; // default
    let statusClass = 'success';
    let statusText = 'Good';
    
    if (riskLevel === 'HIGH') {
      healthPercentage = 30;
      statusClass = 'error';
      statusText = 'Needs Attention';
    } else if (riskLevel === 'MEDIUM') {
      healthPercentage = 60;
      statusClass = 'warning';
      statusText = 'Monitor';
    }
    
    if (healthBar) healthBar.style.width = healthPercentage + '%';
    if (overallStatus) {
      overallStatus.className = `status-${statusClass}`;
      overallStatus.textContent = statusText;
    }
    
    if (summary) {
      summary.innerHTML = `
        <div class="space-y-3">
          <div class="flex justify-between items-center p-3 bg-dark-800/50 rounded-xl">
            <span class="text-dark-300 text-sm">Latest Assessment</span>
            <span class="text-white font-medium">${latest.type === 'voice' ? 'Voice Test' : 'Motor Test'}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-dark-800/50 rounded-xl">
            <span class="text-dark-300 text-sm">Risk Level</span>
            <span class="status-${riskLevel === 'HIGH' ? 'error' : riskLevel === 'MEDIUM' ? 'warning' : 'success'}">${riskLevel || 'Unknown'}</span>
          </div>
          <div class="flex justify-between items-center p-3 bg-dark-800/50 rounded-xl">
            <span class="text-dark-300 text-sm">Confidence</span>
            <span class="text-white font-medium">${latest.confidence ? (latest.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
          </div>
        </div>
      `;
    }
  }

  auth.onAuthStateChanged(async (user) => {
    if (!user) { 
      location.href = '/login'; 
      return; 
    }
    
    // Check demographics completion
    try {
      const idToken = await user.getIdToken();
      const response = await fetch('/demographics/status', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${idToken}`
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        if (!result.demographicsCompleted) {
          location.href = '/demographics';
          return;
        }
      } else {
        location.href = '/demographics';
        return;
      }
    } catch (error) {
      console.error('Error checking demographics status:', error);
      location.href = '/demographics';
      return;
    }
    
    const tbody = document.getElementById('historyRows');
    const emptyState = document.getElementById('emptyState');

    const [voice, spirals] = await Promise.all([loadVoice(user), loadSpirals(user)]);
    const all = [
      ...voice.map(v => ({ type:'voice', ...v })),
      ...spirals.map(s => ({ type:'spiral', ...s })),
    ].sort((a,b) => {
      const ta = a.createdAt?.toMillis ? a.createdAt.toMillis() : 0;
      const tb = b.createdAt?.toMillis ? b.createdAt.toMillis() : 0;
      return tb - ta;
    });

    // Update metrics
    updateMetrics(all);

    if (all.length === 0) { 
      tbody.innerHTML = '';
      if (emptyState) emptyState.classList.remove('hidden');
      return; 
    }

    if (emptyState) emptyState.classList.add('hidden');
    tbody.innerHTML = all.map(r => rowHtml(r)).join('');

    // Update summary
    updateSummary(all);
  });
})();


