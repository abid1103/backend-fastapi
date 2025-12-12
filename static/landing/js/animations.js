// Advanced animations and interactions for BrandInsight

class BrandInsightAnimations {
    constructor() {
        this.initializeAnimations();
        this.setupAdvancedInteractions();
    }

    initializeAnimations() {
        this.setupParallaxEffects();
        this.setupHoverAnimations();
        this.setupLoadingAnimations();
        this.setupChartAnimations();
        this.setupCursorEffects();
    }

    setupParallaxEffects() {
        // Simple parallax for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const parallaxElements = document.querySelectorAll('.parallax');
            
            parallaxElements.forEach(element => {
                const speed = element.getAttribute('data-speed') || 0.5;
                const yPos = -(scrolled * speed);
                element.style.transform = `translateY(${yPos}px)`;
            });
        });
    }

    setupHoverAnimations() {
        // Enhanced hover effects for cards
        const cards = document.querySelectorAll('.feature-card, .testimonial-card');
        
        cards.forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                this.animateCardHover(e.currentTarget);
            });
            
            card.addEventListener('mouseleave', (e) => {
                this.animateCardLeave(e.currentTarget);
            });
        });

        // Button hover effects
        const buttons = document.querySelectorAll('.btn');
        buttons.forEach(button => {
            button.addEventListener('mouseenter', (e) => {
                this.animateButtonHover(e.currentTarget);
            });
            
            button.addEventListener('mouseleave', (e) => {
                this.animateButtonLeave(e.currentTarget);
            });
        });
    }

    animateCardHover(card) {
        card.style.transform = 'translateY(-10px) scale(1.02)';
        card.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
        
        // Add subtle glow effect
        const glow = document.createElement('div');
        glow.className = 'card-glow';
        glow.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at center, rgba(37, 99, 235, 0.1) 0%, transparent 70%);
            border-radius: inherit;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        card.appendChild(glow);
        
        setTimeout(() => {
            glow.style.opacity = '1';
        }, 10);
    }

    animateCardLeave(card) {
        card.style.transform = 'translateY(0) scale(1)';
        card.style.boxShadow = '';
        
        const glow = card.querySelector('.card-glow');
        if (glow) {
            glow.style.opacity = '0';
            setTimeout(() => glow.remove(), 300);
        }
    }

    animateButtonHover(button) {
        button.style.transform = 'translateY(-2px)';
        
        // Create ripple effect
        const ripple = document.createElement('span');
        ripple.className = 'btn-ripple';
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        `;
        
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        
        button.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    animateButtonLeave(button) {
        button.style.transform = 'translateY(0)';
    }

    setupLoadingAnimations() {
        // Loading states for async operations
        this.showLoading = (element) => {
            element.classList.add('loading');
        };

        this.hideLoading = (element) => {
            element.classList.remove('loading');
        };
    }

    setupChartAnimations() {
        // Animate chart bars in dashboard preview
        const chartBars = document.querySelectorAll('.bar');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateChartBars(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        document.querySelectorAll('.chart-bars').forEach(chart => {
            observer.observe(chart);
        });
    }

    animateChartBars(chart) {
        const bars = chart.querySelectorAll('.bar');
        
        bars.forEach((bar, index) => {
            const height = bar.style.height;
            bar.style.height = '0%';
            
            setTimeout(() => {
                bar.style.transition = 'height 1s ease-in-out';
                bar.style.height = height;
            }, index * 100);
        });
    }

    setupCursorEffects() {
        // Custom cursor effects (optional)
        if (window.innerWidth > 768) {
            this.initializeCursorFollower();
        }
    }

    initializeCursorFollower() {
        const follower = document.createElement('div');
        follower.className = 'cursor-follower';
        follower.style.cssText = `
            position: fixed;
            width: 20px;
            height: 20px;
            background: var(--primary);
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            mix-blend-mode: difference;
            transform: translate(-50%, -50%);
            transition: width 0.2s, height 0.2s;
        `;
        document.body.appendChild(follower);

        let mouseX = 0, mouseY = 0;
        let followerX = 0, followerY = 0;

        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        const animateCursor = () => {
            followerX += (mouseX - followerX) * 0.1;
            followerY += (mouseY - followerY) * 0.1;
            
            follower.style.left = `${followerX}px`;
            follower.style.top = `${followerY}px`;
            
            requestAnimationFrame(animateCursor);
        };

        animateCursor();

        // Scale effect on interactive elements
        const interactiveElements = document.querySelectorAll('a, button, .btn, .card');
        
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', () => {
                follower.style.width = '40px';
                follower.style.height = '40px';
            });
            
            el.addEventListener('mouseleave', () => {
                follower.style.width = '20px';
                follower.style.height = '20px';
            });
        });
    }

    setupAdvancedInteractions() {
        this.setupTypewriterEffect();
        this.setupScrollProgress();
        this.setupParticleEffects();
    }

    setupTypewriterEffect() {
        // Typewriter effect for hero title
        const heroTitle = document.querySelector('.hero-title');
        if (heroTitle) {
            const text = heroTitle.textContent;
            heroTitle.textContent = '';
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    heroTitle.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 50);
                }
            };
            
            // Start typing when hero section is in view
            const observer = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) {
                    typeWriter();
                    observer.unobserve(heroTitle);
                }
            });
            
            observer.observe(heroTitle);
        }
    }

    setupScrollProgress() {
        // Scroll progress indicator
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: var(--primary);
            z-index: 1001;
            transition: width 0.1s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const winHeight = window.innerHeight;
            const docHeight = document.documentElement.scrollHeight;
            const scrollTop = window.pageYOffset;
            const scrollPercent = (scrollTop / (docHeight - winHeight)) * 100;
            
            progressBar.style.width = `${scrollPercent}%`;
        });
    }

    setupParticleEffects() {
        // Simple particle effect for hero section
        const heroSection = document.querySelector('.hero');
        if (!heroSection) return;

        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'particles';
        particlesContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
        `;
        heroSection.appendChild(particlesContainer);

        // Create particles
        for (let i = 0; i < 20; i++) {
            this.createParticle(particlesContainer);
        }
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: 4px;
            height: 4px;
            background: var(--primary);
            border-radius: 50%;
            opacity: 0.3;
        `;

        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        particle.style.left = `${posX}%`;
        particle.style.top = `${posY}%`;

        container.appendChild(particle);

        // Animate particle
        this.animateParticle(particle);
    }

    animateParticle(particle) {
        const duration = 3 + Math.random() * 2;
        const delay = Math.random() * 2;

        particle.style.animation = `
            float ${duration}s ease-in-out ${delay}s infinite,
            fade ${duration}s ease-in-out ${delay}s infinite
        `;

        // Add keyframes for animation
        if (!document.querySelector('#particle-animations')) {
            const style = document.createElement('style');
            style.id = 'particle-animations';
            style.textContent = `
                @keyframes float {
                    0%, 100% { transform: translate(0, 0) rotate(0deg); }
                    25% { transform: translate(10px, -10px) rotate(90deg); }
                    50% { transform: translate(0, -20px) rotate(180deg); }
                    75% { transform: translate(-10px, -10px) rotate(270deg); }
                }
                @keyframes fade {
                    0%, 100% { opacity: 0.1; }
                    50% { opacity: 0.6; }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Method for complex page transitions
    async pageTransition(outPage, inPage) {
        // Add your page transition logic here
        console.log('Transitioning from', outPage, 'to', inPage);
    }

    // Method for animated counters with formatting
    animateFormattedCounter(element, target, duration = 2000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }

            // Format number with commas
            element.textContent = Math.floor(current).toLocaleString();
        }, 16);
    }
}

// Initialize animations when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.brandInsightAnimations = new BrandInsightAnimations();
});

// CSS for ripple effect
const rippleStyles = `
@keyframes ripple {
    to {
        transform: scale(4);
        opacity: 0;
    }
}

.btn-ripple {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: scale(0);
    animation: ripple 0.6s linear;
    pointer-events: none;
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = rippleStyles;
document.head.appendChild(styleSheet);