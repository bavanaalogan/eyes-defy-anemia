// Medical Analysis Application JavaScript
class AnemiaAnalyzer {
    constructor() {
        this.uploadedImages = [];
        this.analysisResults = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupNavigation();
        this.setupDragAndDrop();
        this.setupFormValidation();
    }

    setupEventListeners() {
        // File input and upload
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadArea = document.getElementById('uploadArea');
        const analyzeBtn = document.getElementById('analyzeBtn');

        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        uploadArea.addEventListener('click', () => fileInput.click());

        // Analyze button
        analyzeBtn.addEventListener('click', () => this.performAnalysis());

        // Results actions
        const newAnalysisBtn = document.getElementById('newAnalysisBtn');
        const downloadReportBtn = document.getElementById('downloadReportBtn');
        const shareResultsBtn = document.getElementById('shareResultsBtn');

        if (newAnalysisBtn) newAnalysisBtn.addEventListener('click', () => this.startNewAnalysis());
        if (downloadReportBtn) downloadReportBtn.addEventListener('click', () => this.downloadReport());
        if (shareResultsBtn) shareResultsBtn.addEventListener('click', () => this.shareResults());

        // Form validation
        this.setupFormValidation();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links
                navLinks.forEach(l => l.classList.remove('active'));
                
                // Add active class to clicked link
                link.classList.add('active');
                
                // Smooth scroll to section
                const targetId = link.getAttribute('href');
                const targetSection = document.querySelector(targetId);
                
                if (targetSection) {
                    targetSection.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Update active nav on scroll
        window.addEventListener('scroll', () => {
            const sections = document.querySelectorAll('section');
            const scrollPos = window.scrollY + 100;

            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                const sectionId = section.getAttribute('id');

                if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${sectionId}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        });
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        uploadArea.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles(files);
    }

    handleFileSelect(e) {
        const files = e.target.files;
        this.handleFiles(files);
    }

    handleFiles(files) {
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        Array.from(files).forEach(file => {
            if (!validTypes.includes(file.type)) {
                this.showNotification('Please upload only JPG, PNG, or WEBP images.', 'error');
                return;
            }

            if (file.size > maxSize) {
                this.showNotification('File size should be less than 10MB.', 'error');
                return;
            }

            this.uploadedImages.push(file);
            this.displayImagePreview(file);
        });

        this.updateAnalyzeButton();
    }

    displayImagePreview(file) {
        const preview = document.getElementById('imagePreview');
        const reader = new FileReader();

        reader.onload = (e) => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            
            imageItem.innerHTML = `
                <img src="${e.target.result}" alt="Uploaded image">
                <button class="image-remove" onclick="anemiaAnalyzer.removeImage('${file.name}')">
                    <i class="fas fa-times"></i>
                </button>
            `;

            preview.appendChild(imageItem);
        };

        reader.readAsDataURL(file);
    }

    removeImage(fileName) {
        this.uploadedImages = this.uploadedImages.filter(file => file.name !== fileName);
        
        // Remove from preview
        const preview = document.getElementById('imagePreview');
        const imageItems = preview.querySelectorAll('.image-item');
        
        imageItems.forEach(item => {
            const img = item.querySelector('img');
            if (img && img.alt === 'Uploaded image') {
                // Find matching image by comparing src
                item.remove();
            }
        });

        this.updateAnalyzeButton();
    }

    setupFormValidation() {
        const form = document.getElementById('patientForm');
        const inputs = form.querySelectorAll('input[required], select[required]');

        inputs.forEach(input => {
            input.addEventListener('input', () => this.updateAnalyzeButton());
            input.addEventListener('change', () => this.updateAnalyzeButton());
        });
    }

    updateAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const patientName = document.getElementById('patientName').value.trim();
        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;

        const isFormValid = patientName && age && gender;
        const hasImages = this.uploadedImages.length > 0;

        analyzeBtn.disabled = !(isFormValid && hasImages);
    }

    async performAnalysis() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loader = document.getElementById('loader');

        // Start loading state
        analyzeBtn.classList.add('analyzing');
        analyzeBtn.disabled = true;

        try {
            // Simulate analysis process
            await this.simulateAnalysis();
            
            // Generate results
            this.analysisResults = this.generateAnalysisResults();
            
            // Display results
            this.displayResults();
            
            // Navigate to results section
            document.getElementById('results').scrollIntoView({
                behavior: 'smooth'
            });

        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification('Analysis failed. Please try again.', 'error');
        } finally {
            // Reset button state
            analyzeBtn.classList.remove('analyzing');
            analyzeBtn.disabled = false;
        }
    }

    async simulateAnalysis() {
        // Simulate processing time with progress updates
        const steps = [
            'Preprocessing images...',
            'Detecting eye regions...',
            'Analyzing conjunctival pallor...',
            'Calculating hemoglobin levels...',
            'Generating risk assessment...',
            'Finalizing results...'
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.delay(800);
            console.log(steps[i]);
        }
    }

    generateAnalysisResults() {
        const patientData = this.getPatientData();
        
        // Simulate realistic analysis results based on demographics
        const baseHb = this.calculateBaseHemoglobin(patientData);
        const variation = (Math.random() - 0.5) * 3; // ±1.5 g/dL variation
        const hemoglobin = Math.max(6, Math.min(18, baseHb + variation));
        
        const anemiaStatus = this.determineAnemiaStatus(hemoglobin, patientData.gender);
        const confidence = Math.random() * 30 + 70; // 70-100% confidence
        
        return {
            patientData,
            hemoglobin: Number(hemoglobin.toFixed(1)),
            anemiaStatus,
            confidence: Math.round(confidence),
            risks: this.calculateRisks(hemoglobin, anemiaStatus),
            recommendations: this.generateRecommendations(anemiaStatus, hemoglobin, patientData)
        };
    }

    getPatientData() {
        return {
            name: document.getElementById('patientName').value,
            age: parseInt(document.getElementById('age').value),
            gender: document.getElementById('gender').value,
            symptoms: Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
                .map(cb => cb.value)
        };
    }

    calculateBaseHemoglobin(patientData) {
        // Base hemoglobin levels by demographics
        let baseHb = 13.5; // Average baseline

        if (patientData.gender === 'female') {
            baseHb = 12.5;
        } else if (patientData.gender === 'male') {
            baseHb = 14.5;
        }

        // Age adjustments
        if (patientData.age < 12) {
            baseHb -= 1;
        } else if (patientData.age > 65) {
            baseHb -= 0.5;
        }

        return baseHb;
    }

    determineAnemiaStatus(hemoglobin, gender) {
        const normalRanges = {
            male: { min: 13.8, max: 17.2 },
            female: { min: 12.1, max: 15.1 },
            other: { min: 12.5, max: 16.0 }
        };

        const range = normalRanges[gender] || normalRanges.other;

        if (hemoglobin >= range.min) {
            return { status: 'normal', severity: 'none', color: 'normal' };
        } else if (hemoglobin >= range.min - 2) {
            return { status: 'Mild Anemia', severity: 'mild', color: 'mild' };
        } else if (hemoglobin >= range.min - 4) {
            return { status: 'Moderate Anemia', severity: 'moderate', color: 'moderate' };
        } else {
            return { status: 'Severe Anemia', severity: 'severe', color: 'severe' };
        }
    }

    calculateRisks(hemoglobin, anemiaStatus) {
        const risks = {
            mild: 0,
            moderate: 0,
            severe: 0
        };

        switch (anemiaStatus.severity) {
            case 'mild':
                risks.mild = Math.random() * 40 + 60; // 60-100%
                risks.moderate = Math.random() * 30 + 10; // 10-40%
                risks.severe = Math.random() * 15; // 0-15%
                break;
            case 'moderate':
                risks.mild = Math.random() * 20 + 20; // 20-40%
                risks.moderate = Math.random() * 40 + 60; // 60-100%
                risks.severe = Math.random() * 30 + 10; // 10-40%
                break;
            case 'severe':
                risks.mild = Math.random() * 15; // 0-15%
                risks.moderate = Math.random() * 30 + 20; // 20-50%
                risks.severe = Math.random() * 30 + 70; // 70-100%
                break;
            default:
                risks.mild = Math.random() * 20; // 0-20%
                risks.moderate = Math.random() * 10; // 0-10%
                risks.severe = Math.random() * 5; // 0-5%
        }

        return risks;
    }

    generateRecommendations(anemiaStatus, hemoglobin, patientData) {
        const recommendations = [];

        if (anemiaStatus.severity !== 'none') {
            recommendations.push({
                icon: 'fas fa-user-md',
                text: '<strong>Medical Consultation:</strong> Schedule an appointment with your healthcare provider for comprehensive evaluation and proper diagnosis.'
            });

            recommendations.push({
                icon: 'fas fa-vial',
                text: '<strong>Laboratory Tests:</strong> Complete blood count (CBC) and iron studies recommended to determine the underlying cause.'
            });

            if (anemiaStatus.severity === 'mild' || anemiaStatus.severity === 'moderate') {
                recommendations.push({
                    icon: 'fas fa-apple-alt',
                    text: '<strong>Dietary Changes:</strong> Increase iron-rich foods such as lean meats, spinach, legumes, and fortified cereals.'
                });

                recommendations.push({
                    icon: 'fas fa-pills',
                    text: '<strong>Supplements:</strong> Consider iron supplementation as recommended by your healthcare provider.'
                });
            }

            if (anemiaStatus.severity === 'severe') {
                recommendations.push({
                    icon: 'fas fa-exclamation-triangle',
                    text: '<strong>Urgent Care:</strong> Seek immediate medical attention due to severe anemia indication. This may require intensive treatment.'
                });
            }
        } else {
            recommendations.push({
                icon: 'fas fa-heart',
                text: '<strong>Maintain Health:</strong> Continue with a balanced diet rich in iron, vitamin B12, and folate to maintain healthy hemoglobin levels.'
            });

            recommendations.push({
                icon: 'fas fa-calendar-check',
                text: '<strong>Regular Monitoring:</strong> Consider annual health checkups to monitor your blood health status.'
            });
        }

        return recommendations;
    }

    displayResults() {
        const resultsSection = document.getElementById('results');
        resultsSection.style.display = 'block';

        // Update anemia status
        this.updateAnemiaStatus();
        
        // Update hemoglobin level
        this.updateHemoglobinLevel();
        
        // Update risk assessment
        this.updateRiskAssessment();
        
        // Update recommendations
        this.updateRecommendations();

        // Animate the results appearance
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }

    updateAnemiaStatus() {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceValue = document.getElementById('confidenceValue');

        const { anemiaStatus, confidence } = this.analysisResults;

        // Update status
        statusIndicator.className = `status-indicator ${anemiaStatus.color}`;
        statusText.textContent = anemiaStatus.status === 'normal' ? 'Normal' : anemiaStatus.status;

        // Animate confidence meter
        setTimeout(() => {
            confidenceBar.style.width = `${confidence}%`;
            confidenceValue.textContent = `${confidence}%`;
        }, 500);
    }

    updateHemoglobinLevel() {
        const hbValue = document.getElementById('hbValue');
        const normalRange = document.getElementById('normalRange');

        const { hemoglobin, patientData } = this.analysisResults;

        // Animate hemoglobin value
        let current = 0;
        const target = hemoglobin;
        const increment = target / 30;

        const animation = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(animation);
            }
            hbValue.textContent = current.toFixed(1);
        }, 50);

        // Update normal range based on gender
        const ranges = {
            male: '13.8-17.2 g/dL',
            female: '12.1-15.1 g/dL',
            other: '12.5-16.0 g/dL'
        };
        normalRange.textContent = ranges[patientData.gender] || ranges.other;
    }

    updateRiskAssessment() {
        const { risks } = this.analysisResults;

        setTimeout(() => {
            document.getElementById('mildRisk').style.width = `${risks.mild}%`;
        }, 700);

        setTimeout(() => {
            document.getElementById('moderateRisk').style.width = `${risks.moderate}%`;
        }, 900);

        setTimeout(() => {
            document.getElementById('severeRisk').style.width = `${risks.severe}%`;
        }, 1100);
    }

    updateRecommendations() {
        const recommendationsList = document.getElementById('recommendations');
        const { recommendations } = this.analysisResults;

        recommendationsList.innerHTML = '';

        recommendations.forEach((rec, index) => {
            const item = document.createElement('div');
            item.className = 'recommendation-item';
            item.style.opacity = '0';
            item.style.transform = 'translateY(20px)';
            
            item.innerHTML = `
                <i class="${rec.icon} recommendation-icon"></i>
                <div class="recommendation-text">${rec.text}</div>
            `;

            recommendationsList.appendChild(item);

            // Animate appearance
            setTimeout(() => {
                item.style.transition = 'all 0.3s ease';
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, index * 200 + 500);
        });
    }

    startNewAnalysis() {
        // Reset all data
        this.uploadedImages = [];
        this.analysisResults = null;

        // Reset form
        document.getElementById('patientForm').reset();

        // Clear image preview
        document.getElementById('imagePreview').innerHTML = '';

        // Hide results section
        document.getElementById('results').style.display = 'none';

        // Update analyze button
        this.updateAnalyzeButton();

        // Navigate to analysis section
        document.getElementById('analysis').scrollIntoView({ behavior: 'smooth' });

        this.showNotification('Ready for new analysis', 'success');
    }

    async downloadReport() {
        if (!this.analysisResults) {
            this.showNotification('No results to download', 'error');
            return;
        }

        // Generate PDF report (simplified simulation)
        this.showNotification('Generating report...', 'info');

        // Simulate report generation
        await this.delay(2000);

        // Create downloadable content
        const reportContent = this.generateReportContent();
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `anemia-report-${this.analysisResults.patientData.name.replace(/\s+/g, '-')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        this.showNotification('Report downloaded successfully', 'success');
    }

    generateReportContent() {
        const { patientData, hemoglobin, anemiaStatus, confidence, recommendations } = this.analysisResults;
        
        return `
EYES DEFY ANEMIA - MEDICAL ANALYSIS REPORT
==========================================

Patient Information:
- Name: ${patientData.name}
- Age: ${patientData.age}
- Gender: ${patientData.gender}
- Symptoms: ${patientData.symptoms.length ? patientData.symptoms.join(', ') : 'None reported'}

Analysis Results:
- Hemoglobin Level: ${hemoglobin} g/dL
- Status: ${anemiaStatus.status}
- Confidence: ${confidence}%
- Generated: ${new Date().toLocaleString()}

Recommendations:
${recommendations.map(rec => `- ${rec.text.replace(/<[^>]*>/g, '')}`).join('\n')}

MEDICAL DISCLAIMER:
This analysis is for informational purposes only and should not replace 
professional medical advice. Please consult with a healthcare provider 
for proper diagnosis and treatment.

© 2024 Eyes Defy Anemia - Medical Analysis Platform
        `.trim();
    }

    shareResults() {
        if (!this.analysisResults) {
            this.showNotification('No results to share', 'error');
            return;
        }

        const shareText = `Eyes Defy Anemia Analysis Results:\nHemoglobin: ${this.analysisResults.hemoglobin} g/dL\nStatus: ${this.analysisResults.anemiaStatus.status}\nConfidence: ${this.analysisResults.confidence}%`;

        if (navigator.share) {
            navigator.share({
                title: 'Anemia Analysis Results',
                text: shareText,
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(shareText).then(() => {
                this.showNotification('Results copied to clipboard', 'success');
            });
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            font-weight: 500;
            max-width: 300px;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        notification.textContent = message;
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application
const anemiaAnalyzer = new AnemiaAnalyzer();

// Add some interactive enhancements
document.addEventListener('DOMContentLoaded', () => {
    // Add loading animation to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });

    // Add hover effects to cards
    const cards = document.querySelectorAll('.upload-card, .patient-card, .result-card, .about-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'transform 0.3s ease';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.hero-stats, .analysis-grid, .about-grid');
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Add keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        // Close any open modals or reset states
        const resultsSection = document.getElementById('results');
        if (resultsSection.style.display !== 'none') {
            // Could add modal close functionality here
        }
    }
});

// Performance optimization: Lazy load images
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            }
        });
    });

    // Observe images with data-src attribute
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}
