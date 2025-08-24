// Common authentication utilities for VocalScan
(function(){
  window.vsAuth = {
    // Check if user is authenticated and has completed demographics
    async ensureAuthenticatedWithDemographics() {
      return new Promise(resolve => {
        const auth = window.vsFirebase.auth;
        auth.onAuthStateChanged(async user => {
          if (!user) { 
            location.href = '/login'; 
            return;
          }
          
          // Check if demographics are completed
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
              // If we can't check status, redirect to demographics to be safe
              location.href = '/demographics';
              return;
            }
            
            resolve(user);
          } catch (error) {
            console.error('Error checking demographics status:', error);
            location.href = '/demographics';
          }
        });
      });
    },

    // Simple auth check (just checks if user is logged in)
    async ensureAuthenticated() {
      return new Promise(resolve => {
        const auth = window.vsFirebase.auth;
        auth.onAuthStateChanged(user => {
          if (!user) { 
            location.href = '/login'; 
          } else { 
            resolve(user); 
          }
        });
      });
    }
  };
})();
