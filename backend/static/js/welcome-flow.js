// VocalScan Welcome Flow - Guided onboarding for new users
(function(){
  window.vsWelcome = {
    steps: [
      {
        id: 'welcome',
        title: 'Welcome to VocalScan! 🎉',
        content: 'VocalScan helps detect early signs of neurological and respiratory conditions through voice analysis and drawing tests. Let\'s get you started with a quick walkthrough.',
        target: null,
        page: '/dashboard',
        action: 'next'
      },
      {
        id: 'voice-intro',
        title: 'Voice Recording Test',
        content: 'First, we\'ll record your voice saying a specific phrase. This helps us analyze your vocal patterns for potential health indicators.',
        target: null,
        page: '/record',
        action: 'navigate'
      },
      {
        id: 'voice-instructions',
        title: 'Say the Phrase',
        content: 'Click "Start" and clearly say: <br><strong>"The quick brown fox jumps over the lazy dog"</strong><br>Speak naturally and at a comfortable pace.',
        target: '#startBtn',
        page: '/record',
        action: 'wait-for-start'
      },
      {
        id: 'voice-submit',
        title: 'Submit Your Recording',
        content: 'Perfect! Now click "Upload & Analyze" to process your voice recording.',
        target: '#submitBtn',
        page: '/record',
        action: 'hide-and-wait'
      },
      {
        id: 'spiral-intro',
        title: 'Spiral Drawing Test',
        content: 'Next, we\'ll test your motor skills with a spiral drawing. This can help detect tremors or coordination issues.',
        target: null,
        page: '/spiral',
        action: 'navigate'
      },
      {
        id: 'spiral-instructions',
        title: 'Draw a Spiral',
        content: 'Use your mouse or finger to draw a spiral from the center outward. Try to follow the guide lines and keep your movement smooth.',
        target: '#spiralCanvas',
        page: '/spiral',
        action: 'next'
      },
      {
        id: 'spiral-submit',
        title: 'Submit Your Drawing',
        content: 'Excellent! Now click "Save & Analyze" to process your spiral drawing.',
        target: '#submitSpiralBtn',
        page: '/spiral',
        action: 'wait-for-spiral-submit'
      },
      {
        id: 'results-intro',
        title: 'View Your Results',
        content: 'Let\'s check your dashboard to see the results of both tests. This is where you can track your health assessments over time.',
        target: null,
        page: '/dashboard',
        action: 'navigate'
      },
      {
        id: 'dashboard-tour',
        title: 'Your Health Dashboard',
        content: 'Here you can see your test history, risk assessments, and track changes over time. You can always come back here to view past results.',
        target: '#historyTable',
        page: '/dashboard',
        action: 'highlight'
      },
      {
        id: 'complete',
        title: 'Welcome Complete! ✅',
        content: 'Congratulations! You\'ve completed your first VocalScan assessment. You can now:<br>• Record new voice samples<br>• Take spiral drawing tests<br>• Monitor your health trends<br>• Update your profile information',
        target: null,
        page: '/dashboard',
        action: 'exit'
      }
    ],

    currentStep: 0,
    isActive: false,
    overlay: null,
    tooltip: null,
    waitListener: null,
    currentUserId: null,

    async start() {
      console.log('Welcome flow: Starting tour');
      this.isActive = true;
      this.currentStep = 0;
      await this.markWelcomeStarted();
      this.showStep(this.steps[0]);
    },

    async markWelcomeStarted() {
      try {
        const user = window.vsFirebase.auth.currentUser;
        if (user) {
          const idToken = await user.getIdToken();
          const response = await fetch('/demographics', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${idToken}`
            },
            body: JSON.stringify({ welcomeFlowStarted: true })
          });
          
          if (!response.ok) {
            console.log('Welcome flow: Failed to mark as started, continuing anyway');
          }
        }
      } catch (error) {
        console.error('Welcome flow: Error marking as started:', error);
        // Continue anyway - this is not critical
      }
    },

    async markWelcomeCompleted() {
      try {
        const user = window.vsFirebase.auth.currentUser;
        if (user) {
          const idToken = await user.getIdToken();
          const response = await fetch('/demographics', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${idToken}`
            },
            body: JSON.stringify({ welcomeFlowCompleted: true })
          });
          
          if (!response.ok) {
            console.log('Welcome flow: Failed to mark as completed');
          }
        }
      } catch (error) {
        console.error('Welcome flow: Error marking as completed:', error);
      }
    },

    showStep(step) {
      console.log('Welcome flow: Showing step', this.currentStep, step.title);
      this.createOverlay();
      this.createTooltip(step);
      
      if (step.target) {
        this.highlightElement(step.target);
      }

      // Set up automatic progression for wait-for actions
      this.setupWaitForActions(step);
    },

    createOverlay() {
      console.log('Welcome flow: creating overlay');
      if (this.overlay) this.overlay.remove();
      
      this.overlay = document.createElement('div');
      this.overlay.id = 'welcome-overlay';
      this.overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(4px);
        z-index: 9998;
        transition: all 0.3s ease;
        pointer-events: none;
      `;
      document.body.appendChild(this.overlay);
      console.log('Welcome flow: overlay created and added to DOM');
    },

    createTooltip(step) {
      console.log('Welcome flow: creating tooltip for step:', step.title);
      if (this.tooltip) this.tooltip.remove();

      this.tooltip = document.createElement('div');
      this.tooltip.id = 'welcome-tooltip';
      this.tooltip.style.cssText = `
        position: fixed;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        z-index: 9999;
        max-width: 420px;
        font-family: 'Inter', sans-serif;
        transform: translateY(-10px);
        animation: welcomeFadeIn 0.3s ease forwards;
        backdrop-filter: blur(8px);
      `;

      const stepNumber = this.currentStep + 1;
      const totalSteps = this.steps.length;
      
      this.tooltip.innerHTML = `
        <div style="margin-bottom: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-size: 18px; font-weight: 600; color: #f1f5f9;">${step.title}</h3>
            <span style="background: #475569; color: #cbd5e1; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">${stepNumber}/${totalSteps}</span>
          </div>
          <div style="color: #cbd5e1; font-size: 14px; line-height: 1.5;">${step.content}</div>
        </div>
        <div style="display: flex; gap: 12px; justify-content: flex-end;">
          ${this.currentStep > 0 ? '<button id="welcome-prev" style="padding: 8px 16px; border: 1px solid #475569; background: #334155; color: #e2e8f0; border-radius: 8px; cursor: pointer; font-size: 14px; transition: all 0.2s;">Previous</button>' : ''}
          ${step.action === 'no-button' || step.action === 'wait-for-start' || step.action === 'wait-for-spiral-submit' ? '' : step.action === 'exit' ? '<button id="welcome-exit" style="padding: 8px 16px; background: linear-gradient(135deg, #10b981, #059669); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s;">Finish Tour</button>' : step.action === 'hide-and-wait' ? '<button id="welcome-hide" style="padding: 8px 16px; background: linear-gradient(135deg, #f59e0b, #d97706); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s;">Continue</button>' : `<button id="welcome-next" style="padding: 8px 16px; background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.2s;">
            ${step.action === 'navigate' ? 'Continue' : 'Next'}
          </button>`}
          <button id="welcome-skip" style="padding: 8px 16px; border: none; background: transparent; color: #94a3b8; cursor: pointer; font-size: 14px; transition: all 0.2s;">Skip Tour</button>
        </div>
      `;

      // Position tooltip
      if (step.target) {
        const target = document.querySelector(step.target);
        if (target) {
          const rect = target.getBoundingClientRect();
          
          // Special positioning for voice-submit step - position under the button
          if (step.id === 'voice-submit') {
            this.tooltip.style.top = `${rect.bottom + 10}px`;
            this.tooltip.style.left = `${Math.max(20, rect.left)}px`;
          } else {
            this.tooltip.style.top = `${rect.bottom + 20}px`;
            this.tooltip.style.left = `${Math.max(20, rect.left - 200)}px`;
          }
        } else {
          this.centerTooltip();
        }
      } else {
        this.centerTooltip();
      }

      document.body.appendChild(this.tooltip);
      console.log('Welcome flow: tooltip created and added to DOM');

      // Add event listeners
      const nextBtn = document.getElementById('welcome-next');
      const exitBtn = document.getElementById('welcome-exit');
      const hideBtn = document.getElementById('welcome-hide');
      
      if (nextBtn) {
        console.log('Welcome flow: Adding click listener to next button');
        nextBtn.onclick = () => {
          console.log('Welcome flow: Next button clicked');
          this.nextStep();
        };
      } else {
        console.log('Welcome flow: Next button not found');
      }
      
      if (exitBtn) {
        console.log('Welcome flow: Adding click listener to exit button');
        exitBtn.onclick = () => {
          console.log('Welcome flow: Exit button clicked');
          this.complete();
        };
      }
      
      if (hideBtn) {
        console.log('Welcome flow: Adding click listener to hide button');
        hideBtn.onclick = () => {
          console.log('Welcome flow: Hide button clicked');
          this.hideAndWaitForSubmission();
        };
      }
      
      if (document.getElementById('welcome-prev')) {
        document.getElementById('welcome-prev').onclick = () => this.prevStep();
      }
      document.getElementById('welcome-skip').onclick = () => this.skip();

      // Set up automatic progression for wait-for actions
      this.setupWaitForActions(step);

      // Add CSS animation
      if (!document.getElementById('welcome-styles')) {
        const styles = document.createElement('style');
        styles.id = 'welcome-styles';
        styles.textContent = `
          @keyframes welcomeFadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          .welcome-highlight {
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #4f46e5, 0 0 0 6px rgba(79, 70, 229, 0.3) !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
          }
        `;
        document.head.appendChild(styles);
      }
    },

    setupWaitForActions(step) {
      if (!step.action || !step.action.startsWith('wait-for')) return;
      
      console.log('Welcome flow: Setting up wait-for action:', step.action);
      
      // Remove any existing listeners
      if (this.waitListener) {
        this.waitListener.element?.removeEventListener(this.waitListener.event, this.waitListener.handler);
        this.waitListener = null;
      }
      
      if (step.action === 'wait-for-click' && step.waitFor) {
        const element = document.querySelector(step.waitFor);
        if (element) {
          const handler = () => {
            console.log('Welcome flow: Detected click on', step.waitFor);
            setTimeout(() => this.nextStep(), 1000); // Simple delay
          };
          element.addEventListener('click', handler);
          this.waitListener = { element, event: 'click', handler };
          console.log('Welcome flow: Added click listener to', step.waitFor);
        }
      }
      // All other wait-for actions will use manual "Continue" buttons
    },

    setupWaitForActions(step) {
      if (!step.action || !step.action.startsWith('wait-for')) return;
      
      console.log('Welcome flow: Setting up wait-for action:', step.action);
      
      // Remove any existing listeners
      if (this.waitListener) {
        this.waitListener.element?.removeEventListener(this.waitListener.event, this.waitListener.handler);
        this.waitListener = null;
      }
      
      if (step.action === 'wait-for-start' && step.target) {
        const element = document.querySelector(step.target);
        if (element) {
          const handler = () => {
            console.log('Welcome flow: Detected click on Start button');
            setTimeout(() => this.nextStep(), 1000); // Advance after Start is clicked
          };
          element.addEventListener('click', handler);
          this.waitListener = { element, event: 'click', handler };
          console.log('Welcome flow: Added click listener to Start button');
        }
      } else if (step.action === 'wait-for-spiral-submit') {
        const element = document.querySelector('#submitSpiralBtn');
        if (element) {
          const handler = () => {
            console.log('Welcome flow: Detected click on Spiral Submit button');
            setTimeout(() => this.nextStep(), 500); // Advance after spiral submission
          };
          element.addEventListener('click', handler);
          this.waitListener = { element, event: 'click', handler };
          console.log('Welcome flow: Added click listener to Spiral Submit button');
        }
      }
    },

    centerTooltip() {
      this.tooltip.style.top = '50%';
      this.tooltip.style.left = '50%';
      this.tooltip.style.transform = 'translate(-50%, -50%)';
    },

    highlightElement(selector) {
      // Remove previous highlights
      document.querySelectorAll('.welcome-highlight').forEach(el => {
        el.classList.remove('welcome-highlight');
      });

      const element = document.querySelector(selector);
      if (element) {
        element.classList.add('welcome-highlight');
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    },

    async nextStep() {
      console.log('Welcome flow: nextStep called, current step:', this.currentStep);
      const currentStep = this.steps[this.currentStep];
      console.log('Welcome flow: current step data:', currentStep);
      
      if (currentStep.action === 'finish') {
        console.log('Welcome flow: finishing tour');
        await this.complete();
        return;
      }

      // Increment step first
      this.currentStep++;
      console.log('Welcome flow: incremented to step:', this.currentStep);
      
      if (this.currentStep >= this.steps.length) {
        console.log('Welcome flow: reached end of steps, completing');
        await this.complete();
        return;
      }

      const nextStep = this.steps[this.currentStep];
      console.log('Welcome flow: next step data:', nextStep);
      
      if (nextStep.page && window.location.pathname !== nextStep.page) {
        // Navigate to new page and continue tour there
        console.log('Welcome flow: navigating to:', nextStep.page);
        localStorage.setItem('welcomeStep', this.currentStep.toString());
        window.location.href = nextStep.page;
        return;
      }

      // Show next step on same page
      console.log('Welcome flow: showing next step on same page');
      this.showStep(nextStep);
    },

    prevStep() {
      if (this.currentStep > 0) {
        this.currentStep--;
        const prevStep = this.steps[this.currentStep];
        if (prevStep.page && window.location.pathname !== prevStep.page) {
          localStorage.setItem('welcomeStep', this.currentStep.toString());
          window.location.href = prevStep.page;
        } else {
          this.showStep(prevStep);
        }
      }
    },

    async complete() {
      console.log('Welcome flow: Completing tour');
      await this.markWelcomeCompleted();
      this.cleanup();
      localStorage.removeItem('welcomeStep');
      localStorage.setItem('welcomeFlowCompleted', 'true'); // Mark as completed locally
      
      // Show completion message
      alert('Welcome tour completed! You can now use VocalScan to monitor your health. Check your dashboard anytime to view results.');
    },

    skip() {
      if (confirm('Are you sure you want to skip the welcome tour? You can always explore VocalScan on your own.')) {
        this.complete();
      }
    },

    hideAndWaitForSubmission() {
      console.log('Welcome flow: Hiding tour and waiting for submission');
      // Hide the welcome flow UI
      if (this.overlay) this.overlay.style.display = 'none';
      if (this.tooltip) this.tooltip.style.display = 'none';
      
      // Set up listener for submission completion
      const submitBtn = document.querySelector('#submitBtn');
      if (submitBtn) {
        const handler = () => {
          console.log('Welcome flow: Submit detected, advancing to next step');
          // Remove the listener
          submitBtn.removeEventListener('click', handler);
          // Advance to next step and show tour again
          setTimeout(() => {
            this.nextStep();
          }, 2000); // Wait for submission to process
        };
        submitBtn.addEventListener('click', handler, { passive: true });
        console.log('Welcome flow: Added submission listener, tour hidden');
      }
    },

    // Reset welcome flow (for testing or if user wants to restart)
    reset() {
      console.log('Welcome flow: Resetting tour');
      this.cleanup();
      localStorage.removeItem('welcomeStep');
      localStorage.removeItem('welcomeFlowCompleted');
      this.currentStep = 0;
      this.isActive = false;
    },

    // Force start welcome flow (for testing)
    forceStart() {
      console.log('Welcome flow: Force starting tour');
      this.reset();
      this.start();
    },

    // Handle user change - clear localStorage if different user
    handleUserChange(newUserId) {
      if (this.currentUserId && this.currentUserId !== newUserId) {
        console.log('Welcome flow: User changed, clearing localStorage');
        localStorage.removeItem('welcomeStep');
        localStorage.removeItem('welcomeFlowCompleted');
        this.cleanup();
        this.currentStep = 0;
        this.isActive = false;
      }
      this.currentUserId = newUserId;
      
      // If user logged out, clean up everything
      if (!newUserId) {
        console.log('Welcome flow: User logged out, full cleanup');
        localStorage.removeItem('welcomeStep');
        localStorage.removeItem('welcomeFlowCompleted');
        this.cleanup();
        this.currentStep = 0;
        this.isActive = false;
        this.currentUserId = null;
      }
    },

    cleanup() {
      console.log('Welcome flow: cleanup called');
      this.isActive = false;
      
      // Clean up wait listener
      if (this.waitListener) {
        this.waitListener.element?.removeEventListener(this.waitListener.event, this.waitListener.handler, { passive: true });
        this.waitListener = null;
      }
      
      if (this.overlay) this.overlay.remove();
      if (this.tooltip) this.tooltip.remove();
      document.querySelectorAll('.welcome-highlight').forEach(el => {
        el.classList.remove('welcome-highlight');
      });
    },

    // Resume tour if user navigated to a new page
    resumeOnPage() {
      const savedStep = localStorage.getItem('welcomeStep');
      if (savedStep && this.isActive) {
        this.currentStep = parseInt(savedStep);
        const step = this.steps[this.currentStep];
        console.log('Resuming welcome flow at step:', this.currentStep, step?.title);
        console.log('Current page:', window.location.pathname, 'Expected page:', step?.page);
        
        if (step && step.page === window.location.pathname) {
          console.log('Page matches, showing step');
          setTimeout(() => this.showStep(step), 500); // Small delay for page load
        } else {
          console.log('Page mismatch, waiting for correct page or navigation');
          // The step might be for a different page, that's okay
        }
      }
    },

    // Check if user should start welcome flow
    async shouldStart() {
      const user = window.vsFirebase.auth.currentUser;
      if (!user) {
        console.log('Welcome flow: No user logged in');
        return false;
      }

      try {
        const idToken = await user.getIdToken();
        const response = await fetch('/demographics/status', {
          headers: { 'Authorization': `Bearer ${idToken}` }
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Welcome flow: Demographics status:', data);
          return data.demographicsCompleted && !data.welcomeFlowCompleted;
        } else {
          console.log('Welcome flow: Failed to get demographics status');
        }
      } catch (error) {
        console.error('Error checking welcome flow status:', error);
      }
      return false;
    }
  };

  // Auto-start welcome flow for eligible users
  document.addEventListener('DOMContentLoaded', async () => {
    console.log('Welcome flow: DOM loaded');
    // Wait for auth to initialize
    window.vsFirebase.auth.onAuthStateChanged(async (user) => {
      console.log('Welcome flow: Auth state changed, user:', !!user);
      if (user) {
        // Handle user change - clear localStorage if different user
        window.vsWelcome.handleUserChange(user.uid);
        
        const shouldStart = await window.vsWelcome.shouldStart();
        console.log('Welcome flow: Should start?', shouldStart);
        
        if (shouldStart) {
          // Clear any conflicting local storage
          localStorage.removeItem('welcomeFlowCompleted');
          
          // Small delay to ensure page is fully loaded
          setTimeout(() => {
            console.log('Welcome flow: Starting on page:', window.location.pathname);
            // Only start if we're on dashboard AND no tour in progress
            if (window.location.pathname === '/dashboard' && !localStorage.getItem('welcomeStep')) {
              window.vsWelcome.start();
            } else if (localStorage.getItem('welcomeStep')) {
              // Resume existing tour
              console.log('Welcome flow: Tour in progress, resuming');
              window.vsWelcome.isActive = true;
              window.vsWelcome.resumeOnPage();
            }
          }, 1000);
        } else if (localStorage.getItem('welcomeStep') && !localStorage.getItem('welcomeFlowCompleted')) {
          // Resume existing tour
          console.log('Welcome flow: Resuming existing tour');
          window.vsWelcome.isActive = true;
          window.vsWelcome.resumeOnPage();
        }
      } else {
        // User logged out - clear everything
        console.log('Welcome flow: User logged out, clearing state');
        window.vsWelcome.handleUserChange(null);
      }
    });
  });
})();
