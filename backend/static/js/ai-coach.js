// AI Coach JavaScript
class AICoach {
    constructor() {
        this.geminiApiKey = 'AIzaSyDFfdgjhuRUSWU_QNtz6a_zfSjawZqGtMI';
        this.geminiApiUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';
        this.currentPlan = null;
        
        this.initializeElements();
        this.bindEvents();
        this.loadStoredFeedback();
    }

    initializeElements() {
        this.conditionInput = document.getElementById('condition-input');
        this.getPlanBtn = document.getElementById('get-plan-btn');
        this.loadingState = document.getElementById('loading-state');
        this.suggestedPlan = document.getElementById('suggested-plan');
        this.planContent = document.getElementById('plan-content');
        this.errorState = document.getElementById('error-state');
        this.errorMessage = document.getElementById('error-message');
        this.successFeedback = document.getElementById('success-feedback');
        
        // Feedback buttons
        this.doItBtn = document.getElementById('do-it-btn');
        this.tooEasyBtn = document.getElementById('too-easy-btn');
        this.tooHardBtn = document.getElementById('too-hard-btn');
        
        // Save button
        this.savePlanBtn = document.getElementById('save-plan-btn');
    }

    bindEvents() {
        // Enable/disable button based on input
        this.conditionInput.addEventListener('input', () => {
            this.getPlanBtn.disabled = !this.conditionInput.value.trim();
        });

        // Get plan button
        this.getPlanBtn.addEventListener('click', () => {
            this.getExercisePlan();
        });

        // Feedback buttons
        this.doItBtn.addEventListener('click', () => this.submitFeedback('do_it'));
        this.tooEasyBtn.addEventListener('click', () => this.submitFeedback('too_easy'));
        this.tooHardBtn.addEventListener('click', () => this.submitFeedback('too_hard'));
        
        // Save plan button
        this.savePlanBtn.addEventListener('click', () => this.savePlanToProfile());

        // Enter key to submit
        this.conditionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.getPlanBtn.disabled) {
                    this.getExercisePlan();
                }
            }
        });
    }

    async getExercisePlan() {
        const userInput = this.conditionInput.value.trim();
        if (!userInput) return;

        this.showLoading();
        this.hideError();
        this.hideSuccess();

        try {
            // Get user demographics first
            let demographics = null;
            try {
                const user = firebase.auth().currentUser;
                if (user) {
                    const idToken = await user.getIdToken();
                    const response = await fetch('/demographics/data', {
                        headers: {
                            'Authorization': `Bearer ${idToken}`
                        }
                    });
                    
                    if (response.ok) {
                        const userData = await response.json();
                        if (userData.demographicsCompleted) {
                            demographics = userData;
                        }
                    }
                }
            } catch (error) {
                console.log('Could not fetch demographics:', error);
                // Continue without demographics if there's an error
            }

            const systemPrompt = `You are a specialized AI exercise coach EXCLUSIVELY for Parkinson's disease patients. This is a Parkinson's-focused platform, so all recommendations must be specifically tailored for Parkinson's disease management.

**Your Role:**
- Create personalized exercise plans specifically for Parkinson's disease patients
- Focus on Parkinson's-specific symptoms and challenges
- Address the unique needs of Parkinson's patients
- Keep exercises safe, gentle, and achievable for Parkinson's patients

**Parkinson's-Specific Focus Areas:**
- **Bradykinesia (slowness of movement)** - exercises to improve speed and coordination
- **Rigidity (muscle stiffness)** - gentle stretching and range of motion exercises
- **Tremor management** - stability and fine motor skill exercises
- **Postural instability** - balance and core strengthening exercises
- **Gait disturbances** - walking and stepping exercises
- **Freezing episodes** - movement initiation exercises
- **Voice and speech issues** - breathing and vocal exercises
- **Fine motor skills** - hand and finger coordination exercises

**Safety Guidelines for Parkinson's:**
- Always include safety considerations for fall prevention
- Emphasize slow, deliberate movements
- Include rest periods between exercises
- Recommend exercises that can be done seated if needed
- Focus on functional movements that improve daily activities
- Consider medication timing and "on/off" periods
- Include caregiver support recommendations where appropriate

**Format Requirements:**
- Use **bold** for section headers
- Use bullet points (-) for individual exercises
- Include specific time recommendations
- Add safety notes where relevant
- Keep exercises simple and achievable
- Include modifications for different ability levels

**Patient Context:**
${demographics ? `
- Age: ${demographics.age || 'Not specified'}
- Gender: ${demographics.gender || 'Not specified'}
- Medical History: ${demographics.medicalHistory || 'Not specified'}
- Current Symptoms: ${demographics.symptoms ? demographics.symptoms.join(', ') : 'Not specified'}
- Medications: ${demographics.medications ? demographics.medications.join(', ') : 'None specified'}
- Exercise Level: ${demographics.lifestyle?.exercise || 'Not specified'}
` : '- No demographic information available'}

Make each exercise plan specifically tailored to Parkinson's disease management and symptom improvement. Consider the patient's specific symptoms, age, and current condition.`;

            const requestBody = {
                contents: [{
                    parts: [{
                        text: `${systemPrompt}\n\nPatient input: ${userInput}`
                    }]
                }],
                generationConfig: {
                    temperature: 0.7,
                    maxOutputTokens: 500,
                }
            };

            console.log('Sending request to Gemini API:', requestBody);

            // Try the API call
            let response;
            try {
                response = await fetch(`${this.geminiApiUrl}?key=${this.geminiApiKey}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                });
            } catch (fetchError) {
                console.error('Fetch error:', fetchError);
                // If fetch fails, provide a fallback response for testing
                const fallbackPlan = this.generateFallbackPlan(userInput);
                this.currentPlan = {
                    plan: fallbackPlan,
                    userInput: userInput,
                    timestamp: new Date().toISOString()
                };
                this.displayPlan(fallbackPlan);
                return;
            }

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('API Error Response:', errorText);
                throw new Error(`API request failed: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('Gemini API Response:', data);
            
            if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts && data.candidates[0].content.parts[0]) {
                const plan = data.candidates[0].content.parts[0].text;
                this.currentPlan = {
                    plan: plan,
                    userInput: userInput,
                    timestamp: new Date().toISOString()
                };
                this.displayPlan(plan);
            } else {
                console.error('Unexpected response structure:', data);
                throw new Error('Invalid response format from Gemini API');
            }

        } catch (error) {
            console.error('Error getting exercise plan:', error);
            
            // Show more specific error message
            if (error.message.includes('API request failed: 403')) {
                this.showError('API key error. Please check the configuration.');
            } else if (error.message.includes('API request failed: 429')) {
                this.showError('Rate limit exceeded. Please try again in a moment.');
            } else if (error.message.includes('Failed to fetch')) {
                this.showError('Network error. Please check your internet connection.');
            } else {
                this.showError(`Failed to generate exercise plan: ${error.message}`);
            }
        } finally {
            this.hideLoading();
        }
    }

    displayPlan(plan) {
        // Format the plan with better styling
        const formattedPlan = this.formatExercisePlan(plan);
        this.planContent.innerHTML = formattedPlan;
        this.suggestedPlan.classList.remove('hidden');
        this.suggestedPlan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    formatExercisePlan(plan) {
        // Split the plan into sections
        const lines = plan.split('\n');
        let formattedHTML = '';
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            if (line.startsWith('**') && line.endsWith('**')) {
                // Section headers (bold text)
                const sectionTitle = line.replace(/\*\*/g, '');
                formattedHTML += `
                    <div class="mb-4">
                        <div class="flex items-center mb-3">
                            <div class="w-2 h-2 bg-primary-500 rounded-full mr-3"></div>
                            <h4 class="text-lg font-semibold text-white">${sectionTitle}</h4>
                        </div>
                    </div>
                `;
            } else if (line.match(/^\d+\./)) {
                // Numbered exercises
                const exerciseText = line.replace(/^\d+\.\s*/, '');
                formattedHTML += `
                    <div class="ml-6 mb-3 p-3 bg-dark-600/30 rounded-lg border-l-4 border-primary-500">
                        <p class="text-white font-medium">${exerciseText}</p>
                    </div>
                `;
            } else if (line.startsWith('-')) {
                // Bullet points
                const bulletText = line.replace(/^-\s*/, '');
                formattedHTML += `
                    <div class="ml-6 mb-2 flex items-start">
                        <div class="w-1.5 h-1.5 bg-primary-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                        <p class="text-dark-200">${bulletText}</p>
                    </div>
                `;
            } else if (line.length > 0) {
                // Regular text
                formattedHTML += `<p class="text-dark-200 mb-3">${line}</p>`;
            } else {
                // Empty lines
                formattedHTML += '<div class="h-2"></div>';
            }
        }
        
        return formattedHTML;
    }

    submitFeedback(feedbackType) {
        if (!this.currentPlan) return;

        const feedback = {
            ...this.currentPlan,
            feedback: feedbackType,
            feedbackTimestamp: new Date().toISOString()
        };

        // Store in localStorage
        localStorage.setItem('aiCoachLastFeedback', JSON.stringify(feedback));

        // Show success message at bottom
        this.showSuccess();
        
        // Clear input for next use
        this.conditionInput.value = '';
        this.getPlanBtn.disabled = true;
        
        // Hide success message after 3 seconds
        setTimeout(() => {
            this.hideSuccess();
        }, 3000);
    }

    loadStoredFeedback() {
        const stored = localStorage.getItem('aiCoachLastFeedback');
        if (stored) {
            try {
                const feedback = JSON.parse(stored);
                console.log('Last feedback:', feedback);
                // Could be used to improve future recommendations
            } catch (error) {
                console.error('Error parsing stored feedback:', error);
            }
        }
    }

    showLoading() {
        this.loadingState.classList.remove('hidden');
        this.getPlanBtn.disabled = true;
    }

    hideLoading() {
        this.loadingState.classList.add('hidden');
        this.getPlanBtn.disabled = !this.conditionInput.value.trim();
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorState.classList.remove('hidden');
    }

    hideError() {
        this.errorState.classList.add('hidden');
    }

    showSuccess() {
        this.successFeedback.classList.remove('hidden');
    }

    hideSuccess() {
        this.successFeedback.classList.add('hidden');
    }

    async savePlanToProfile() {
        if (!this.currentPlan) return;

        try {
            // Get current user
            const user = firebase.auth().currentUser;
            if (!user) {
                this.showError('Please log in to save your exercise plan.');
                return;
            }

            const planData = {
                plan: this.currentPlan.plan,
                userInput: this.currentPlan.userInput,
                timestamp: this.currentPlan.timestamp,
                savedAt: new Date().toISOString(),
                type: 'exercise_plan'
            };

            // Save to Firestore
            const db = firebase.firestore();
            await db.collection('users').doc(user.uid).collection('savedPlans').add(planData);

            // Show success message
            this.showSaveSuccess();
            
            // Update button text temporarily
            const originalText = this.savePlanBtn.innerHTML;
            this.savePlanBtn.innerHTML = `
                <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Saved!
            `;
            this.savePlanBtn.disabled = true;
            
            // Reset button after 3 seconds
            setTimeout(() => {
                this.savePlanBtn.innerHTML = originalText;
                this.savePlanBtn.disabled = false;
            }, 3000);

        } catch (error) {
            console.error('Error saving plan:', error);
            this.showError('Failed to save plan. Please try again.');
        }
    }

    showSaveSuccess() {
        // Create a temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        successDiv.innerHTML = `
            <div class="flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                Exercise plan saved to your profile!
            </div>
        `;
        
        document.body.appendChild(successDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.parentNode.removeChild(successDiv);
            }
        }, 3000);
    }

    generateFallbackPlan(userInput) {
        const plans = {
            'stiff': `Based on your Parkinson's rigidity (muscle stiffness), here's a specialized exercise plan:

**Morning Range of Motion (8-10 minutes)**
- Gentle neck rotations - turn head slowly side to side (5 reps each direction)
- Shoulder shrugs and rolls to reduce upper body stiffness (10 reps)
- Arm circles - start small, gradually increase size (5 forward, 5 backward)
- Hip circles while standing near support (5 each direction)
- Ankle and wrist rotations (10 each direction)

**Balance and Posture (7 minutes)**
- Stand near wall, practice shifting weight foot to foot (30 seconds each side)
- Gentle side-to-side swaying with support (1 minute)
- Heel-to-toe standing (tandem stance) with support (30 seconds)
- Sit-to-stand exercises (3-5 repetitions with 30-second rest between)

**Gait Training (10 minutes)**
- Slow, deliberate steps with heel-to-toe pattern (2 minutes)
- Practice turning while walking (wide turns) - 5 turns each direction
- Use walking aids if needed for safety
- Focus on arm swing coordination (2 minutes)

**Breathing and Relaxation (5 minutes)**
- Deep breathing with hand on belly (10 breaths)
- Progressive muscle relaxation (3 minutes)
- Gentle stretching while seated (2 minutes)

**Safety Notes:** Always have support nearby, move slowly, and stop if you feel dizzy or unsteady. Consider doing exercises during your "on" medication periods for best results.`,

            'shaky': `For Parkinson's tremor management, here's a specialized exercise plan:

**Tremor Control Exercises (10 minutes)**
- Finger tapping on table - start slow, increase speed gradually (2 minutes)
- Thumb to each finger touch - focus on precision (2 minutes)
- Gentle hand squeezes with soft ball or stress ball (2 minutes)
- Wrist circles - small, controlled movements (2 minutes)
- Hand stabilization exercises - rest hands on table, practice lifting fingers individually (2 minutes)

**Arm Stability Training (8 minutes)**
- Rest arms on table, palms down, practice lifting fingers one by one (2 minutes)
- Gentle arm raises to shoulder level with controlled movement (2 minutes)
- Wall push-ups (standing) for arm strength (2 minutes)
- Shoulder blade squeezes to improve posture (2 minutes)

**Fine Motor Skills (7 minutes)**
- Pick up small objects (coins, buttons) with precision (2 minutes)
- Practice writing with larger movements and weighted pens (2 minutes)
- Use weighted utensils if helpful for daily activities (1 minute)
- Button and zipper practice with larger items (2 minutes)

**Relaxation and Breathing (5 minutes)**
- Deep breathing exercises to reduce stress (2 minutes)
- Progressive muscle relaxation (2 minutes)
- Gentle hand and wrist stretches (1 minute)

**Safety Notes:** Take breaks as needed, don't rush movements, and use support if needed for balance. Consider using weighted utensils and adaptive equipment for daily activities.`,

            'tired': `For Parkinson's fatigue management, here's a specialized energy-boosting plan:

**Gentle Movement Initiation (5 minutes)**
- Slow marching in place with arm coordination (2 minutes)
- Gentle arm swings to improve mobility (1 minute)
- Shoulder rolls to reduce stiffness (1 minute)
- Ankle pumps while seated (1 minute)

**Seated Strength Building (10 minutes)**
- Ankle circles and foot exercises (2 minutes)
- Knee lifts while seated (alternating legs) (2 minutes)
- Gentle torso twists with arm movement (2 minutes)
- Arm raises to shoulder level with controlled movement (2 minutes)
- Seated leg extensions (2 minutes)

**Energy and Breathing (5 minutes)**
- Deep breathing exercises to increase oxygen flow (2 minutes)
- Gentle stretching while seated (1 minute)
- Progressive muscle relaxation (1 minute)
- Mental imagery exercises (1 minute)

**Rest and Recovery**
- Take 2-3 minute breaks between exercise sets
- Stay hydrated throughout the day
- Listen to your body's energy levels
- Practice energy conservation techniques

**Safety Notes:** It's okay to do less if you're very tired. Quality over quantity. Stop if you feel dizzy or overly fatigued. Consider doing exercises during your "on" medication periods when you have more energy.`
        };

        // Find the best matching plan based on keywords
        const input = userInput.toLowerCase();
        if (input.includes('stiff') || input.includes('rigid')) {
            return plans.stiff;
        } else if (input.includes('shaky') || input.includes('tremor') || input.includes('hand')) {
            return plans.shaky;
        } else if (input.includes('tired') || input.includes('fatigue') || input.includes('exhausted')) {
            return plans.tired;
        } else {
            // Default plan
            return `Based on your Parkinson's symptoms, here's a comprehensive exercise plan:

**Warm-up and Mobility (8 minutes)**
- Gentle stretching focusing on stiff areas (2 minutes)
- Deep breathing exercises (2 minutes)
- Slow walking in place with arm coordination (2 minutes)
- Range of motion exercises for neck, shoulders, and hips (2 minutes)

**Balance and Posture (10 minutes)**
- Standing near support, practice weight shifting (2 minutes)
- Heel-to-toe standing (tandem stance) with support (2 minutes)
- Gentle side-to-side movements (2 minutes)
- Sit-to-stand exercises (3-5 repetitions with 30-second rest between) (4 minutes)

**Strength and Coordination (7 minutes)**
- Seated leg lifts with controlled movement (2 minutes)
- Arm raises with coordination (2 minutes)
- Gentle resistance exercises using body weight (2 minutes)
- Fine motor skill exercises (1 minute)

**Cool-down and Relaxation (5 minutes)**
- Gentle stretching while seated (2 minutes)
- Progressive muscle relaxation (2 minutes)
- Deep breathing exercises (1 minute)

**Safety Notes:** Always have support nearby, move slowly and deliberately, and stop if you feel dizzy or unsteady. Focus on quality of movement over quantity. Consider doing exercises during your "on" medication periods for best results.`;
        }
    }
}

// Initialize AI Coach when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AICoach();
});
